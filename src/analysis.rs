//! 音频指标计算。判定逻辑集中在这里, 界面只负责调用。
//!
//! 与 Python 版 check_audio.py 的差别:
//!   - 底噪改成"整段音频静音部分的能量平均", 不再用分位数, 也不再只在人声轨判定
//!   - 不校验 meta json, 只看音频本身

use std::path::{Path, PathBuf};

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ---- CSV 表头 (交付方规定, 顺序不要动) --------------------------------------
pub const COLUMNS: [&str; 15] = [
    "文件名", "格式", "采样率", "时长", "码率kbps", "底噪dBFS", "截至频率Khz",
    "人声活动比例", "伴奏活动比例", "人声伴奏分贝差", "峰值电平", "平均幅值", "是否削波",
    "是否满足要求", "备注（如有）",
];
// "是否满足要求"/"备注" 永远是最后两列, 用相对位置算下标 —— 以后再插新列
// 不用满仓库找哪里写死了 10/11, 那正是这次踩过的坑。
pub const VERDICT_COL: usize = COLUMNS.len() - 2;
pub const NOTES_COL: usize = COLUMNS.len() - 1;

// ---- 判定阈值 (需要标定时改这里) --------------------------------------------
pub const INS_ACTIVITY: f64 = 0.80;      // 伴奏活动比例 >=
pub const VOC_ACTIVITY: f64 = 0.40;      // 人声活动比例 >=
pub const RMS_DIFF_LO: f64 = -15.0;      // 人声-伴奏 平均RMS差 dB, 下限
pub const RMS_DIFF_HI: f64 = 10.0;       // 上限
pub const CUTOFF_HZ: f64 = 15000.0;      // 截止频率 >=
pub const BITRATE_KBPS: f64 = 320.0;     // 码率 >=
pub const NOISE_DBFS: f64 = -40.0;       // 底噪 <
pub const RT60_S: f64 = 0.30;            // RT60 <
pub const PEAK_DB_MAX: f64 = -1.0;       // 峰值电平: 不超过这个值(没有下限)
pub const AVG_DB_LO: f64 = -8.0;         // 平均幅值(活动段 RMS): 下限
pub const AVG_DB_HI: f64 = -3.0;         // 上限
// 削波判定阈值。这里踩过一个坑, 记录一下取舍:
//
// 第一版按"贴不贴这个文件自己的峰值"(相对阈值)算, 想解决"削波发生在数字化之前
// (话筒前级过载), 到 ADC 时还留着几个 dB 余量, 波形已经削平但采样值没到满量程"
// 这种漏检。结果在真实素材上大批量误报: 任何平滑波峰(不管削没削波)在接近顶点
// 时导数天然趋近于零, 连续好几个采样贴着"这段波形自己的峰值"是数学必然, 频率
// 越低(贝斯、鼓)越明显, 跟削波毫无关系。改完之后正常母带(混音轨习惯贴近 0dB
// 是行业惯例)反而被大量错判, 比如一首正常歌从 0 处误报炸到 4000+ 处。
//
// 现在退回绝对阈值, 只是比原来的 0.999(-0.0087dB)略微放宽到 -0.5dB, 给"前级
// 削波但数字域还有点余量"的常见情况留一点容差。代价是: 如果削波发生得更早、
// 后面又被大幅降过增益, 峰值远低于 0dB, 这种更极端的情况还是测不出来 —— 两难
// 之间选了误报率更低的一边。
const CLIP_THRESHOLD: f64 = 0.9441;      // -0.5dBFS
const CLIP_MIN_RUN: usize = 3;           // 连续触顶达到这个采样数才算削波; 1~2个可能只是自然的峰值瞬间

// ---- 分析参数 ---------------------------------------------------------------
const WIN: f64 = 0.020;                  // 包络分析窗 (秒)
const HOP: f64 = 0.010;                  // 步进 (秒)
const ACT_REL_DB: f64 = 35.0;            // 活动判定: 高于本轨 P95 电平 - 35 dB
const ACT_ABS_DBFS: f64 = -55.0;         // 且高于该绝对电平
const SILENCE_DBFS: f64 = -90.0;         // 低于此视为数字静音(剪辑留白), 不算本底噪声
const MIN_SILENCE_S: f64 = 0.5;          // 静音不足这么久就别报底噪了, 样本太少不可信
const MIN_DURATION_S: f64 = 1.0;         // 短于此的文件没有分析价值

const CUT_NFFT: usize = 8192;
const CUT_MAX_FRAMES: usize = 300;
const CUT_MIN_HZ: f64 = 8000.0;          // 只在 8kHz 以上找"砖墙"
const CUT_SPAN_HZ: f64 = 1000.0;         // 跌落观察跨度
const CUT_DROP_DB: f64 = 25.0;           // 跨度内跌落超过该值 = 编码截止

const RT60_HEAD: f64 = 5.0;              // T20 拟合区间: 峰下 5 ~ 25 dB
const RT60_TAIL: f64 = 25.0;
const RT60_MAX_FIT_S: f64 = 0.8;         // 拟合段过长 = 其实是停顿, 不是混响
const RT60_MIN_R2: f64 = 0.90;           // 拟合优度门槛, 挡掉"衰减穿过停顿"的伪段
const RT60_MIN_SEGS: usize = 3;          // 有效衰减段少于此数则判定为无法估计

pub const AUDIO_EXT: [&str; 9] = [
    "wav", "mp3", "flac", "m4a", "aac", "ogg", "wma", "aiff", "aif",
];

// ============================================================ 解码
pub struct Decoded {
    pub mono: Vec<f64>,
    pub sr: u32,
    pub channels: usize,
    /// 原始采样的最大绝对值(下混增益归一之前)。峰值电平必须用这个, 不能用
    /// `mono` —— mono 为了对齐响度被乘过增益, 拿它算峰值会是缩放过的假峰值。
    pub peak: f64,
    /// 削波事件数: 某个声道连续 >=CLIP_MIN_RUN 个采样触顶算一次, 不是采样计数
    /// (否则一段长削波会把数字撑得没有意义)。同理必须用原始采样, 不能用 mono。
    pub clip_events: usize,
}

/// 解码成单声道 f64。下混按能量归一 —— 直接取平均的话, 左右不相关的立体声轨
/// 会比单声道轨系统性低读约 3dB, 而人声常是单声道、伴奏常是立体声, 那 3dB 会
/// 直接算进"人声伴奏分贝差"里。
pub fn decode(path: &Path) -> Result<Decoded, String> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::codecs::CodecParameters;
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path).map_err(|e| format!("打不开文件({e})"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|_| "认不出这个音频格式(可能不是音频文件, 或者文件已损坏)".to_string())?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or("文件里没有音频轨")?;
    let track_id = track.id;
    let params = match track.codec_params.as_ref() {
        Some(CodecParameters::Audio(a)) => a.clone(),
        _ => return Err("文件里没有音频轨".into()),
    };

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("没有对应的解码器({e})"))?;

    let mut mono: Vec<f64> = Vec::new();
    let mut sum_sq_all = 0.0f64;      // 所有声道所有采样的平方和, 用于能量归一
    let mut n_all = 0usize;
    let mut peak = 0.0f64;             // 原始采样最大绝对值, 增益归一之前记录
    let mut sr = params.sample_rate.unwrap_or(0);
    let mut channels = params.channels.as_ref().map(|c| c.count()).unwrap_or(1);
    let mut inter: Vec<f32> = Vec::new();
    let mut clip_run: Vec<usize> = Vec::new();   // 每个声道各自的"当前连续触顶计数"
    let mut clip_events = 0usize;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(_) => break,          // 尾部损坏: 用已经解出来的部分, 不整首作废
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                let spec = buf.spec();
                sr = spec.rate();
                channels = spec.channels().count().max(1);
                if clip_run.len() != channels {
                    clip_run = vec![0usize; channels];   // 声道数变了(极少见)就重新起算
                }
                inter.clear();
                buf.copy_to_vec_interleaved(&mut inter);
                for fr in inter.chunks(channels) {
                    let mut s = 0.0f64;
                    for (ch, &v) in fr.iter().enumerate() {
                        let v = v as f64;
                        s += v;
                        sum_sq_all += v * v;
                        peak = peak.max(v.abs());
                        if clip_step(&mut clip_run[ch], v, CLIP_THRESHOLD) {
                            clip_events += 1;
                        }
                    }
                    n_all += fr.len();
                    mono.push(s / fr.len() as f64);
                }
            }
            Err(SymError::DecodeError(_)) => continue,   // 单帧坏了跳过
            Err(_) => break,
        }
    }

    if sr == 0 {
        return Err("采样率无效".into());
    }
    if mono.is_empty() {
        return Err("解不出任何音频数据(文件可能损坏或为空)".into());
    }

    let mono_ms: f64 = mono.iter().map(|v| v * v).sum::<f64>() / mono.len() as f64;
    if mono_ms > 0.0 && n_all > 0 {
        let gain = (sum_sq_all / n_all as f64).sqrt() / mono_ms.sqrt();
        if gain.is_finite() && gain > 0.0 {
            for v in mono.iter_mut() {
                *v *= gain;
            }
        }
    }
    Ok(Decoded { mono, sr, channels, peak, clip_events })
}

// ============================================================ 指标
/// 逐帧 RMS 电平 (dBFS)。用平方前缀和算, 避免展开成大矩阵。
pub fn frame_db(x: &[f64], sr: u32) -> Vec<f64> {
    let n = (sr as f64 * WIN) as usize;
    let h = (sr as f64 * HOP) as usize;
    if n == 0 || h == 0 || x.len() < n {
        return Vec::new();
    }
    let mut c = Vec::with_capacity(x.len() + 1);
    c.push(0.0f64);
    let mut acc = 0.0f64;
    for &v in x {
        acc += v * v;
        c.push(acc);
    }
    (0..=(x.len() - n))
        .step_by(h)
        .map(|s| 10.0 * ((c[s + n] - c[s]) / n as f64 + 1e-20).log10())
        .collect()
}

/// 与 numpy.percentile 一致的线性插值分位数。
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64)
    }
}

/// 活动判定门限: 高于 本轨P95-35dB 与 -55dBFS 中较高者。
fn active_threshold(db: &[f64]) -> f64 {
    let mut s = db.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (percentile(&s, 95.0) - ACT_REL_DB).max(ACT_ABS_DBFS)
}

pub fn active_seconds(db: &[f64]) -> f64 {
    if db.is_empty() {
        return 0.0;
    }
    let thr = active_threshold(db);
    db.iter().filter(|&&v| v > thr).count() as f64 * HOP
}

/// 底噪 = 整段音频静音部分的平均电平(按能量平均)。
///
/// 静音部分 = 活动判定门限以下的帧, 但要排除数字静音(剪辑留的绝对零, 不是本底噪声)。
/// 静音总时长不足 MIN_SILENCE_S 就返回 None —— 连续演奏的轨根本没有静音段可测,
/// 这时报出来的数字只会是"最安静的乐句", 不是底噪。
pub fn noise_floor_db(db: &[f64]) -> Option<f64> {
    if db.is_empty() {
        return None;
    }
    let thr = active_threshold(db);
    let sil: Vec<f64> = db
        .iter()
        .copied()
        .filter(|&v| v <= thr && v > SILENCE_DBFS)
        .collect();
    if sil.len() as f64 * HOP < MIN_SILENCE_S {
        return None;
    }
    let mean_pow = sil.iter().map(|v| 10f64.powf(v / 10.0)).sum::<f64>() / sil.len() as f64;
    Some(10.0 * mean_pow.log10())
}

pub fn rms_db(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    10.0 * (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-20).log10()
}

/// 平均幅值(RMS, dB), 只算活动段的采样。"活动/静音"的判断标准复用
/// active_threshold(简单的能量阈值 VAD): 跟活动比例、底噪同一套逻辑,
/// 不然安静的前奏尾奏、句间停顿会被一起摊进整段平均, 拉低读数,
/// 显得录音比实际更"温"。
///
/// db 是调用方已经算好的逐帧包络(frame_db 的结果), 这里复用, 不重复算一遍;
/// 每个活跃帧只计入 HOP 那部分采样(不是整个 WIN 窗口), 跟 active_seconds 数
/// 时长的口径保持一致 —— 帧与帧之间有重叠, 按 WIN 算会把重叠区间重复计数。
fn active_rms_db(x: &[f64], sr: u32, db: &[f64]) -> f64 {
    if db.is_empty() {
        return rms_db(x);   // 太短测不出包络, 退回整段算
    }
    let thr = active_threshold(db);
    let hop = (sr as f64 * HOP) as usize;
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;
    for (k, &d) in db.iter().enumerate() {
        if d <= thr {
            continue;
        }
        let s = k * hop;
        let e = (s + hop).min(x.len());
        sum_sq += x[s..e].iter().map(|v| v * v).sum::<f64>();
        n += e - s;
    }
    if n == 0 {
        return rms_db(x);   // 极端情况全曲没有一帧判定为活跃, 保险退回整段算
    }
    10.0 * (sum_sq / n as f64 + 1e-20).log10()
}

/// 峰值电平(dBFS) = 20*log10(最大绝对采样值)。真静音(peak=0)时没有意义, 返回 None。
fn peak_dbfs(peak: f64) -> Option<f64> {
    if peak > 0.0 { Some(20.0 * peak.log10()) } else { None }
}

/// 单声道的单个采样触顶检测: run 是调用方持有的"该声道当前连续触顶计数",
/// threshold 是绝对幅度门槛(见 CLIP_THRESHOLD 上面那段注释, 记录了为什么最终
/// 选了绝对阈值而不是相对这个文件自己峰值算)。连续触顶数刚好达到 CLIP_MIN_RUN
/// 时返回 true(新增一次削波事件), 之后同一段继续触顶不重复计数。只认"连续
/// 多个采样顶到满量程附近"这种硬削波; 孤立 1~2 个触顶采样可能只是正常的瞬时
/// 峰值, 不算削波。
fn clip_step(run: &mut usize, sample: f64, threshold: f64) -> bool {
    if sample.abs() >= threshold {
        *run += 1;
        *run == CLIP_MIN_RUN
    } else {
        *run = 0;
        false
    }
}

/// 长时平均谱 (dB), 只取有内容的帧, 最多 CUT_MAX_FRAMES 帧。
fn ltas_db(x: &[f64]) -> Option<Vec<f64>> {
    let n = CUT_NFFT;
    if x.len() < n * 2 {
        return None;
    }
    let starts: Vec<usize> = (0..x.len() - n).step_by(n / 2).collect();
    let energy: Vec<f64> = starts
        .iter()
        .map(|&s| x[s..s + n].iter().map(|v| v * v).sum::<f64>())
        .collect();
    let mut sorted = energy.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = percentile(&sorted, 50.0);
    let kept: Vec<usize> = starts
        .iter()
        .zip(&energy)
        .filter(|(_, e)| **e > med)
        .map(|(&s, _)| s)
        .collect();
    if kept.is_empty() {
        return None;
    }
    // 均匀抽样, 与 numpy.linspace 取整一致
    let used: Vec<usize> = if kept.len() > CUT_MAX_FRAMES {
        (0..CUT_MAX_FRAMES)
            .map(|i| {
                let t = i as f64 * (kept.len() - 1) as f64 / (CUT_MAX_FRAMES - 1) as f64;
                kept[t as usize]
            })
            .collect()
    } else {
        kept
    };

    let hann: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut psd = vec![0.0f64; n / 2 + 1];
    let mut buf = vec![Complex::<f64>::new(0.0, 0.0); n];
    for &s in &used {
        for i in 0..n {
            buf[i] = Complex::new(x[s + i] * hann[i], 0.0);
        }
        fft.process(&mut buf);
        for (k, p) in psd.iter_mut().enumerate() {
            *p += buf[k].norm_sqr();
        }
    }
    let cnt = used.len() as f64;
    let raw: Vec<f64> = psd.iter().map(|p| 10.0 * (p / cnt + 1e-20).log10()).collect();

    // 9 点滑动平均, 边界按零填充 (与 numpy.convolve(..., 'same') 一致)
    let w = 9usize;
    let half = w / 2;
    let smooth: Vec<f64> = (0..raw.len())
        .map(|i| {
            let mut s = 0.0;
            for k in 0..w {
                let j = i as isize + k as isize - half as isize;
                if j >= 0 && (j as usize) < raw.len() {
                    s += raw[j as usize];
                }
            }
            s / w as f64
        })
        .collect();
    Some(smooth)
}

/// 截止频率 = 8kHz 以上第一处"砖墙"(1kHz 内跌落 >=25dB)的半功率点; 没有砖墙就是满带宽。
/// 不用"峰值 -X dB"的判法: 干声本身高频就低, 那样会把好文件判成 4kHz。
pub fn cutoff_hz(x: &[f64], sr: u32) -> Option<f64> {
    let n = CUT_NFFT;
    let db = ltas_db(x)?;
    let w = ((CUT_SPAN_HZ * n as f64 / sr as f64) as usize).max(1);
    let lo = (CUT_MIN_HZ * n as f64 / sr as f64) as usize;
    if lo + w >= db.len() {
        return Some(sr as f64 / 2.0);
    }
    for i in lo..db.len() - w {
        if db[i] - db[i + w] >= CUT_DROP_DB {
            // 砖墙的半功率点才是截止频率, 跌落窗起点会比真截止低约 1kHz
            let mid = (db[i] + db[i + w]) / 2.0;
            let mut last = i;
            for j in i..=(i + w).min(db.len() - 1) {
                if db[j] >= mid {
                    last = j;
                }
            }
            return Some(last as f64 * sr as f64 / n as f64);
        }
    }
    Some(sr as f64 / 2.0)   // 没有砖墙 = 满带宽
}

/// 乐句衰减段做 T20 线性外推 RT60, 取中位数。
///
/// 没有冲激响应可用, 这是估计值: 只保留拟合优度 R^2>0.9、时长合理、且整段都在底噪
/// 之上的衰减段 —— 否则拟合会横穿乐句之间的停顿, 算出 3~30 秒的假混响。
pub fn rt60(db: &[f64], noise: Option<f64>) -> Option<Rt60> {
    if db.len() < 50 {
        return None;
    }
    let peak = db.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let loud = peak - 25.0;
    let floor = noise.unwrap_or(SILENCE_DBFS) + 5.0;
    let n = db.len();
    let mut vals: Vec<f64> = Vec::new();
    let mut i = 1usize;

    while i < n - 1 {
        if !(db[i] >= loud && db[i] >= db[i - 1] && db[i] >= db[i + 1]) {
            i += 1;
            continue;
        }
        let pk = db[i];
        let mut j = i + 1;
        while j < n && db[j] > pk - RT60_HEAD {
            j += 1;
        }
        let mut k = j;
        let mut ok = true;
        while k < n && db[k] > pk - RT60_TAIL {
            if db[k] > pk {           // 中途又起声, 放弃本段
                ok = false;
                break;
            }
            k += 1;
        }
        let next = k.max(i + 1);
        if ok
            && j + 2 < k
            && k < n
            && (k - j) as f64 * HOP <= RT60_MAX_FIT_S
            && pk - RT60_TAIL >= floor
        {
            let m = k - j;
            let xs: Vec<f64> = (0..m).map(|t| (j + t) as f64 * HOP).collect();
            let ys = &db[j..k];
            let mx = xs.iter().sum::<f64>() / m as f64;
            let my = ys.iter().sum::<f64>() / m as f64;
            let sxx: f64 = xs.iter().map(|v| (v - mx) * (v - mx)).sum();
            let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| (a - mx) * (b - my)).sum();
            if sxx > 0.0 {
                let slope = sxy / sxx;
                let intercept = my - slope * mx;
                let ss_res: f64 = xs
                    .iter()
                    .zip(ys)
                    .map(|(a, b)| {
                        let e = b - (slope * a + intercept);
                        e * e
                    })
                    .sum();
                let ss_tot: f64 = ys.iter().map(|b| (b - my) * (b - my)).sum();
                let r2 = 1.0 - ss_res / ss_tot.max(1e-9);
                if slope < -1.0 && r2 >= RT60_MIN_R2 {
                    vals.push(-60.0 / slope);
                }
            }
        }
        i = next;
    }

    if vals.len() < RT60_MIN_SEGS {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = vals.len();
    let median = if m % 2 == 1 {
        vals[m / 2]
    } else {
        (vals[m / 2 - 1] + vals[m / 2]) / 2.0
    };
    Some(Rt60 {
        median,
        segments: m,
        over_limit: vals.iter().filter(|&&v| v >= RT60_S).count(),
    })
}

pub struct Rt60 {
    pub median: f64,
    pub segments: usize,
    /// 超过限值的段数。段与段之间本来就散(实测常差 10 倍), 所以别看离散度,
    /// 要看有多少段真的超标 —— 合格的歌通常 0~33%, 判不合格的那首是 60%。
    pub over_limit: usize,
}

impl Rt60 {
    /// 衰减段太少, 中位数证据不足, 不该单凭它退货。实测合格样本多在 18~58 段。
    pub fn is_thin(&self) -> bool {
        self.segments < 10
    }
}

// ============================================================ 单文件分析
pub struct Track {
    pub path: PathBuf,
    pub sr: u32,
    pub dur: f64,
    pub bitrate: f64,
    pub active_s: f64,
    pub rms: f64,
    pub noise: Option<f64>,
    pub cutoff: Option<f64>,
    pub rt60: Option<Rt60>,
    pub peak_db: Option<f64>,
    pub clip_events: usize,
}

pub fn analyse(path: &Path) -> Result<Track, String> {
    let size = std::fs::metadata(path).map_err(|e| format!("读不到文件信息({e})"))?.len();
    if size == 0 {
        return Err("文件是空的(0 字节)".into());
    }
    let d = decode(path)?;
    let dur = d.mono.len() as f64 / d.sr as f64;
    if dur < MIN_DURATION_S {
        return Err(format!("音频只有 {dur:.2} 秒, 太短, 测不出有效指标"));
    }
    let db = frame_db(&d.mono, d.sr);
    let noise = noise_floor_db(&db);
    Ok(Track {
        path: path.to_path_buf(),
        sr: d.sr,
        dur,
        // 码率用 文件字节数/时长 实测: 对 wav 和 mp3 都成立, 且 VBR 拿到的是真实均值
        bitrate: size as f64 * 8.0 / dur / 1000.0,
        active_s: active_seconds(&db),
        rms: active_rms_db(&d.mono, d.sr, &db),
        noise,
        cutoff: cutoff_hz(&d.mono, d.sr),
        rt60: rt60(&db, noise),
        // 峰值电平/削波都必须用 d.peak/d.clip_events (原始采样, 增益归一之前), 不能用 d.mono
        peak_db: peak_dbfs(d.peak),
        clip_events: d.clip_events,
    })
}

// ============================================================ 角色 / 分组
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Mix,
    Voc,
    Ins,
}

// 角色后缀按长到短排, 免得 "vocals" 被 "voc" 先吃掉
const VOC_SUFFIX: [&str; 4] = ["vocals", "vocal", "voc", "人声"];
const INS_SUFFIX: [&str; 4] = ["instrumental", "ins", "acc", "伴奏"];
const VOC_BRACKET: [&str; 3] = ["vocal", "vocals", "人声"];
const INS_BRACKET: [&str; 4] = ["instrumental", "inst", "off vocal", "伴奏"];

fn strip_bracket(stem: &str, tokens: &[&str]) -> Option<String> {
    let low = stem.to_ascii_lowercase();   // 只降 ASCII, 保证字节偏移与原串一致
    for t in tokens {
        for (o, c) in [('(', ')'), ('[', ']')] {
            let pat = format!("{o}{t}{c}");
            if let Some(pos) = low.find(&pat) {
                let mut s = String::from(&stem[..pos]);
                s.push_str(&stem[pos + pat.len()..]);
                return Some(s.trim_matches([' ', '_', '-']).to_string());
            }
        }
    }
    None
}

/// 把文件名拆成 (歌名, 角色)。
///
/// 角色只认结尾后缀(-Voc / _Ins / -人声)或带括号的标记((Instrumental))。
/// 不能用"名字里含 ins"来判 —— 一整个文件夹的 xxx_(Instrumental)(1).mp3 会被
/// 全判成同一首歌的伴奏轨, 只剩一条; 而 Insane / Wins 这种词也会被误伤。
pub fn split_role(stem: &str) -> (String, Role) {
    let low = stem.to_ascii_lowercase();
    for (tokens, role) in [(&VOC_SUFFIX[..], Role::Voc), (&INS_SUFFIX[..], Role::Ins)] {
        for t in tokens {
            if low.ends_with(t) {
                let base = &stem[..stem.len() - t.len()];
                let trimmed = base.trim_end_matches([' ', '_', '-']);
                // 拉丁词要求前面有分隔符(避免 Wins 结尾被当成 ins); 中文标记不要求
                if !t.is_ascii() || trimmed.len() < base.len() {
                    return (trimmed.to_string(), role);
                }
            }
        }
    }
    if let Some(b) = strip_bracket(stem, &VOC_BRACKET) {
        return (b, Role::Voc);
    }
    if let Some(b) = strip_bracket(stem, &INS_BRACKET) {
        return (b, Role::Ins);
    }
    (stem.to_string(), Role::Mix)
}

/// 一组待校验的音频 = 一首歌。
pub struct SongGroup {
    pub title: String,
    pub files: Vec<(PathBuf, Role)>,
}

/// 把一批音频按歌名分组。一个目录既可能是一首歌的 1~3 条轨, 也可能是一堆互不
/// 相干的单曲(一整个文件夹的伴奏), 靠歌名分组区分。
pub fn group_audio(files: &[PathBuf]) -> Vec<SongGroup> {
    let mut order: Vec<String> = Vec::new();
    let mut songs: std::collections::HashMap<String, Vec<(PathBuf, Role)>> = Default::default();
    let mut extras: Vec<(PathBuf, Role)> = Vec::new();

    for p in files {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let (base, role) = split_role(stem);
        let slot = songs.entry(base.clone()).or_insert_with(|| {
            order.push(base.clone());
            Vec::new()
        });
        if slot.iter().any(|(_, r)| *r == role) {
            extras.push((p.clone(), role));   // 同名同角色撞车, 单独成组, 保证每个文件都出结果
        } else {
            slot.push((p.clone(), role));
        }
    }

    let mut out: Vec<SongGroup> = order
        .into_iter()
        .map(|t| {
            let mut files = songs.remove(&t).unwrap_or_default();
            files.sort_by(|a, b| a.0.cmp(&b.0));
            SongGroup { title: t, files }
        })
        .collect();
    for (p, r) in extras {
        let title = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        out.push(SongGroup { title, files: vec![(p, r)] });
    }
    out
}

// ============================================================ 判定 / 出行
fn fmt_opt(v: Option<f64>, n: usize) -> String {
    match v {
        Some(v) if v.is_finite() => format!("{v:.*}", n),
        _ => String::new(),
    }
}

fn mmss(s: f64) -> String {
    let t = s.round() as i64;
    format!("{:02}:{:02}", t / 60, t % 60)
}

/// 校验一首歌, 每个音频出一行。root 用于把文件名显示成相对路径。
pub fn check_song(group: &SongGroup, root: Option<&Path>) -> Vec<Vec<String>> {
    let mut tracks: Vec<(Role, Result<Track, String>)> = Vec::new();
    for (p, role) in &group.files {
        tracks.push((*role, analyse(p)));
    }

    let get = |want: Role| -> Option<&Track> {
        tracks
            .iter()
            .find(|(r, t)| *r == want && t.is_ok())
            .and_then(|(_, t)| t.as_ref().ok())
    };
    let voc = get(Role::Voc);
    let ins = get(Role::Ins);
    let mix = get(Role::Mix);

    let dur = tracks
        .iter()
        .filter_map(|(_, t)| t.as_ref().ok())
        .map(|t| t.dur)
        .fold(0.0f64, f64::max);

    let voc_act = voc.filter(|_| dur > 0.0).map(|t| t.active_s / dur);
    let ins_act = ins.filter(|_| dur > 0.0).map(|t| t.active_s / dur);
    let diff = match (voc, ins) {
        (Some(v), Some(i)) => Some(v.rms - i.rms),
        _ => None,
    };
    // 混响优先按人声轨算, 其次混音, 只有伴奏轨就用伴奏
    let rev = voc.or(mix).or(ins).and_then(|t| t.rt60.as_ref());

    // 整首歌共用的判定, 会写进这首歌的每一行
    let mut song_fails: Vec<String> = Vec::new();
    let mut song_notes: Vec<String> = Vec::new();
    if let Some(v) = ins_act {
        if v < INS_ACTIVITY {
            song_fails.push(format!("伴奏活动比例{:.1}%<80%", v * 100.0));
        }
    }
    if let Some(v) = voc_act {
        if v < VOC_ACTIVITY {
            song_fails.push(format!("人声活动比例{:.1}%<40%", v * 100.0));
        }
    }
    if let Some(v) = diff {
        if !(RMS_DIFF_LO..=RMS_DIFF_HI).contains(&v) {
            song_fails.push(format!("人声伴奏分贝差{v:+.1}dB超出[-15,10]"));
        }
    }
    // 备注只写"异常和额外信息", 正常情况下应该短到一眼扫过。
    match rev {
        Some(v) => {
            // 括号里点明含义: 看表的人未必知道 RT60 是什么
            song_notes.push(format!("RT60 {:.2}s(混响时间)", v.median));
            if v.median >= RT60_S {
                // 证据薄的时候提一句, 免得靠一个"看着很确定"的数字去退货
                let weak = if v.is_thin() {
                    format!("(仅{}段, 建议复核)", v.segments)
                } else {
                    String::new()
                };
                song_fails.push(format!("混响时间RT60={:.2}s>=0.3s{weak}", v.median));
            }
        }
        None => song_notes.push("RT60未测出(混响时间)".into()),
    }
    // 空着的那几列已经说明"没测", 这里只需点出是缺哪条轨, 不用把指标名列一遍
    let miss: Vec<&str> = [("人声轨", voc.is_none()), ("伴奏轨", ins.is_none())]
        .iter()
        .filter(|(_, m)| *m)
        .map(|(n, _)| *n)
        .collect();
    if !miss.is_empty() {
        song_notes.push(format!("无{}", miss.join("和")));
    }

    let mut rows = Vec::new();
    for ((_, res), (path, _)) in tracks.iter().zip(&group.files) {
        let name = match root {
            Some(r) => path.strip_prefix(r).unwrap_or(path).to_string_lossy().to_string(),
            None => path.file_name().unwrap_or_default().to_string_lossy().to_string(),
        };
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_uppercase();

        let t = match res {
            Err(e) => {
                // 单个文件坏掉不影响同一首歌的其他文件
                let mut row = vec![String::new(); COLUMNS.len()];
                row[0] = name;
                row[1] = ext;
                row[VERDICT_COL] = format!("否：{e}");
                rows.push(row);
                continue;
            }
            Ok(t) => t,
        };

        let mut fails = song_fails.clone();
        let mut notes = song_notes.clone();
        if t.bitrate < BITRATE_KBPS {
            fails.push(format!("码率{:.0}kbps<320", t.bitrate));
        }
        match t.cutoff {
            None => notes.push("截止频率未测出".into()),
            Some(c) if c < CUTOFF_HZ => {
                fails.push(format!("截止频率{:.1}kHz<15kHz", c / 1000.0))
            }
            Some(_) => {}
        }
        match t.noise {
            None => notes.push("底噪未测(无静音段)".into()),
            Some(nf) if nf >= NOISE_DBFS => fails.push(format!("底噪{nf:.1}dBFS高于-40")),
            Some(_) => {}
        }
        match t.peak_db {
            None => notes.push("峰值电平未测出".into()),
            Some(pk) if pk > PEAK_DB_MAX => {
                fails.push(format!("峰值电平{pk:.1}dBFS超过{PEAK_DB_MAX}dBFS"))
            }
            Some(_) => {}
        }
        // 平均幅值 = 活动段 RMS(t.rms), 跟峰值电平是两回事: 峰值管"顶没顶到头",
        // 平均幅值管"整体响不响"——同一个峰值下, 平均幅值越高说明动态压得越死。
        if !(AVG_DB_LO..=AVG_DB_HI).contains(&t.rms) {
            fails.push(format!("平均幅值{:.1}dBFS超出[{AVG_DB_LO},{AVG_DB_HI}]", t.rms));
        }
        // 削波是硬性禁止项, 跟峰值范围分开报: 峰值超标只是"响", 削波是"failed 已经失真"。
        // 一个文件出现削波时峰值必然也贴着 0dBFS、峰值检查本来就会一起不合格,
        // 这里单独给个明确原因, 免得被当成只是"没控制好响度"这种轻微问题。
        if t.clip_events > 0 {
            fails.push(format!("检测到削波({}处连续触顶采样)", t.clip_events));
        }

        rows.push(vec![
            name,
            ext,
            t.sr.to_string(),
            mmss(t.dur),
            format!("{:.0}", t.bitrate),
            fmt_opt(t.noise, 1),
            fmt_opt(t.cutoff.map(|c| c / 1000.0), 2),
            voc_act.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_default(),
            ins_act.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_default(),
            fmt_opt(diff, 1),
            fmt_opt(t.peak_db, 1),
            format!("{:.1}", t.rms),
            if t.clip_events > 0 { format!("是({}处)", t.clip_events) } else { "否".into() },
            if fails.is_empty() { "是".into() } else { format!("否：{}", fails.join("；")) },
            notes.join("；"),
        ]);
    }
    rows
}

// ============================================================ 自检
#[cfg(test)]
mod tests {
    use super::*;

    fn sine(sr: u32, secs: f64, freq: f64, amp: f64) -> Vec<f64> {
        (0..(sr as f64 * secs) as usize)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin())
            .collect()
    }

    // 简易可复现噪声, 免得为了测试引入 rand
    fn noise(n: usize, amp: f64, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                u * amp
            })
            .collect()
    }

    #[test]
    fn activity_ratio() {
        let sr = 48000;
        let mut x = sine(sr, 5.0, 440.0, 0.3);
        x.extend(noise(5 * sr as usize, 1e-4, 7));
        let ratio = active_seconds(&frame_db(&x, sr)) / 10.0;
        assert!((0.48..0.52).contains(&ratio), "活动比例 {ratio}");
    }

    #[test]
    fn active_rms_ignores_silence() {
        let sr = 48000;
        // 5秒响的正弦 + 5秒近乎数字静音的尾巴
        let mut x = sine(sr, 5.0, 440.0, 0.5);
        x.extend(noise(5 * sr as usize, 1e-4, 42));
        let db = frame_db(&x, sr);
        let whole = rms_db(&x);
        let active = active_rms_db(&x, sr, &db);
        let sine_only = rms_db(&sine(sr, 5.0, 440.0, 0.5));
        assert!(
            (active - sine_only).abs() < 1.0,
            "只算活动段应接近纯响段自身RMS, active={active} sine_only={sine_only}"
        );
        assert!(
            active > whole + 2.0,
            "活动段RMS应明显高于被静音拉低的整段RMS, active={active} whole={whole}"
        );
    }

    #[test]
    fn noise_floor_is_mean_of_silence() {
        // 3 秒 -60dBFS 白噪(静音段) + 3 秒大声正弦; 底噪应约等于 -60
        let sr = 48000;
        let mut x = noise(3 * sr as usize, 1e-3 * 3f64.sqrt(), 11); // rms ≈ 1e-3 = -60dBFS
        x.extend(sine(sr, 3.0, 440.0, 0.3));
        let nf = noise_floor_db(&frame_db(&x, sr)).expect("应该测得出底噪");
        assert!((nf - -60.0).abs() < 3.0, "底噪 {nf}");
    }

    #[test]
    fn no_silence_means_no_noise_floor() {
        // 全程有声: 测不出底噪, 必须返回 None 而不是硬报一个数
        let sr = 48000;
        let x = sine(sr, 5.0, 440.0, 0.3);
        assert!(noise_floor_db(&frame_db(&x, sr)).is_none());
    }

    #[test]
    fn cutoff_detects_brickwall_and_fullband() {
        let sr = 48000;
        let n = 20 * sr as usize;
        let x = noise(n, 0.3, 3);
        assert_eq!(cutoff_hz(&x, sr), Some(sr as f64 / 2.0), "满带宽应为奈奎斯特");

        // 12kHz 砖墙低通: 频域截断再逆变换
        let mut planner = FftPlanner::<f64>::new();
        let len = 1 << 19;
        let fwd = planner.plan_fft_forward(len);
        let inv = planner.plan_fft_inverse(len);
        let mut buf: Vec<Complex<f64>> =
            x.iter().take(len).map(|&v| Complex::new(v, 0.0)).collect();
        buf.resize(len, Complex::new(0.0, 0.0));
        fwd.process(&mut buf);
        let cut = (12000.0 * len as f64 / sr as f64) as usize;
        for k in cut..=len - cut {
            buf[k] = Complex::new(0.0, 0.0);
        }
        inv.process(&mut buf);
        let low: Vec<f64> = buf.iter().map(|c| c.re / len as f64).collect();
        let c = cutoff_hz(&low, sr).expect("应测得截止频率");
        assert!((c - 12000.0).abs() < 500.0, "截止频率 {c}");
    }

    #[test]
    fn rt60_matches_known_decay() {
        // 已知 T60=0.25s 的指数衰减脉冲串
        let sr = 48000;
        let mut x: Vec<f64> = Vec::new();
        for r in 0..8 {
            let burst = noise((0.7 * sr as f64) as usize, 1.0, 100 + r);
            for (i, v) in burst.iter().enumerate() {
                let t = i as f64 / sr as f64;
                x.push(v * 10f64.powf(-3.0 * t / 0.25));
            }
            x.extend(noise((0.3 * sr as f64) as usize, 1e-4, 200 + r));
        }
        let db = frame_db(&x, sr);
        let r = rt60(&db, noise_floor_db(&db)).expect("应测得 RT60");
        assert!((r.median - 0.25).abs() < 0.05, "RT60 {}", r.median);
        // 同一个已知衰减, 各段之间不该差太多
        assert!(r.segments >= 3, "衰减段数 {}", r.segments);
    }

    #[test]
    fn peak_level() {
        // 0.5 幅度 -> -6.02 dBFS
        let p = peak_dbfs(0.5).expect("应测得峰值电平");
        assert!((p - (-6.0206)).abs() < 0.01, "峰值电平 {p}");
        // 真静音测不出峰值, 不能硬报 -inf
        assert!(peak_dbfs(0.0).is_none());
        // 满幅 = 0 dBFS
        assert!((peak_dbfs(1.0).unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn clip_detection() {
        let mut run = 0usize;
        let mut events = 0usize;
        // 触顶2次(不够3连续, 不算) / 触顶3连续(算1次) / 触顶4连续(还是只算1次, 不重复计数)
        for v in [0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0] {
            if clip_step(&mut run, v, CLIP_THRESHOLD) {
                events += 1;
            }
        }
        assert_eq!(events, 2, "应识别出两段独立削波(第2/3段各算1次)");

        // 孤立触顶(哪怕全曲仅此一处)不该被当成削波
        let mut run = 0usize;
        let mut events = 0usize;
        for v in [0.3, 1.0, 0.2, 1.0, 0.1] {
            if clip_step(&mut run, v, CLIP_THRESHOLD) {
                events += 1;
            }
        }
        assert_eq!(events, 0, "孤立触顶采样不算削波");

        // -0.5dB 阈值应该比原来的 0.999(-0.0087dB) 松: 贴近但没完全触顶满量程
        // 的连续采样也该算削波, 这是这次放宽阈值要解决的场景。
        let mut run = 0usize;
        let mut events = 0usize;
        for v in [0.5, 0.95, 0.95, 0.95, 0.5] {
            if clip_step(&mut run, v, CLIP_THRESHOLD) {
                events += 1;
            }
        }
        assert_eq!(events, 1, "贴近但没完全顶满的连续采样也该算削波");
    }

    #[test]
    fn role_and_grouping() {
        assert_eq!(split_role("下雨天-Voc"), ("下雨天".into(), Role::Voc));
        assert_eq!(split_role("下雨天_伴奏"), ("下雨天".into(), Role::Ins));
        assert_eq!(split_role("x_(Instrumental)(1)").1, Role::Ins);
        assert_eq!(split_role("Insane Wins").1, Role::Mix);   // 普通词里的 ins 不算

        // 一首歌的三条轨合成一组
        let files: Vec<PathBuf> = ["下雨天.wav", "下雨天-Voc.wav", "下雨天-Ins.wav"]
            .iter()
            .map(PathBuf::from)
            .collect();
        let g = group_audio(&files);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].files.len(), 3);

        // 一整个文件夹的散装伴奏要各成一组
        let files: Vec<PathBuf> = (0..10)
            .map(|i| PathBuf::from(format!("{i}_歌手 - 歌名{i}_(Instrumental)(1).mp3")))
            .collect();
        let g = group_audio(&files);
        assert_eq!(g.len(), 10, "散装伴奏应各成一组");
        assert!(g.iter().all(|s| s.files.len() == 1));
    }
}
