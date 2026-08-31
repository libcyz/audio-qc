//! 原始录音素材校验 (项目B) 的指标与判定。
//!
//! 校验项 (交付方规格):
//!   1. 文件格式必须是 WAV
//!   2. 通道数: 立体声
//!   3. 采样率 >= 44.1kHz
//!   4. 单条时长 >= 1 分钟, 静音段 <= 20%
//!   5. 信噪比 >= 70dB, 频响 20Hz-20kHz ±3dB
//!   6. 未经后续算法处理或滤波, 未压缩
//!
//! 解码和 DSP 全部借项目A 的 `audio_qc::analysis` —— 那一层只测不判, 判定全在
//! 本文件, 跟项目A 的 `check_song` 没有任何关系, 两边的阈值可以各改各的。
//!
//! 第 5、6 条能做到什么程度, 见文件末尾 `局限` 一节, 别当成实验室指标用。

use std::path::Path;

use audio_qc::analysis::{
    active_rms_db, active_seconds, cutoff_hz, decode, energy_mask, frame_db, ltas_db,
    noise_floor_db_above, percentile, CUT_NFFT, HOP,
};

// ---- CSV 表头 ---------------------------------------------------------------
pub const COLUMNS: [&str; 11] = [
    "文件名", "文件类型", "是否立体声", "声道数", "采样率", "时长",
    "静音比例", "信噪比dB", "频响",
    "是否满足要求", "备注（如有）",
];
// 交付方指定的字段是"文件类型 是否立体声 声道数 采样率 时长 静音比例 信噪比dB
// 频响 是否满足要求"这九项。另外两列是加上去的:
//   文件名 —— 没有它整张表对不上是哪个文件, 不算多余字段;
//   备注   —— 频响 ±3dB 的口径、双单声道、疑似限幅这些必须有地方说, 否则只剩
//             一个光秃秃的"否"没法复核。要去掉的话把这一列删了即可, 判定不受影响。
// 跟项目A 一样的约定: "是否满足要求"/"备注" 永远是最后两列, 用相对位置算下标。
pub const VERDICT_COL: usize = COLUMNS.len() - 2;
pub const NOTES_COL: usize = COLUMNS.len() - 1;

// ---- 判定阈值 ---------------------------------------------------------------
pub const REQ_CHANNELS: usize = 2;       // 立体声
pub const MIN_SR: u32 = 44100;           // 采样率 >=
pub const MIN_DUR_S: f64 = 60.0;         // 单条时长 >= 1 分钟
pub const MAX_SILENCE: f64 = 0.20;       // 静音段 <=
pub const MIN_SNR_DB: f64 = 70.0;        // 信噪比 >=
pub const BAND_HI_HZ: f64 = 20000.0;     // 频响上限: 高频要够到这里
pub const BAND_LO_HZ: f64 = 20.0;        // 频响下限
pub const FLATNESS_DB: f64 = 3.0;        // ±3dB, 只有拿测试信号测才有意义, 见"局限"

/// 低于本电平的帧算"绝对数字零", 不计入底噪。
///
/// 项目A 用的是 -90dBFS(音乐母带里剪辑挖的静音), 但校验 70dB 信噪比时那个门限
/// 太高: 底噪本来就该在 -90 附近, 拿 -90 去滤等于把最该统计的帧全扔掉, 信噪比
/// 会系统性偏低。-120 低于任何真实转换器的本底(24bit 理论 -144, 实际 ADC 约
/// -110~-120), 掉到它下面的只可能是数字静音或被静音处理过的段落。
pub const DIGITAL_ZERO_DBFS: f64 = -120.0;

/// 双单声道判据: 侧信号能量占比低于此值就认为左右完全相同。
/// 真立体声这个值在 0.01~0.3 量级, 1e-6 相当于 -60dB, 不会误伤"很窄但确实有宽度"的素材。
const DUAL_MONO_SIDE: f64 = 1e-6;
/// 峰值贴到这个电平以上就提一句"疑似限幅/归一化"。
const LIMITED_PEAK_DB: f64 = -0.1;
/// 砖墙跌落点低于奈奎斯特的这个比例 = 疑似有损压缩来源。
/// 正常录音在奈奎斯特附近也有抗混叠滤波器造成的陡降, 所以不能只看"有没有砖墙"。
const LOSSY_NYQUIST_RATIO: f64 = 0.90;
/// 低频段比中低频参考段低这么多才提"疑似高通滤波" —— 门限放得高, 因为人声素材
/// 本来 20~40Hz 就没什么内容, 门限低了会天天误报。
const LOW_ROLLOFF_DB: f64 = 25.0;
/// 内部数字零段超过这么久才提"疑似静音处理"(掐头去尾的零不算)。
const ZERO_RUN_S: f64 = 0.3;

/// 界面上能改的参数。其余阈值写死成常量, 只把可能要现场标定的放出来。
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Settings {
    pub min_snr_db: f64,
    pub max_silence: f64,
    pub min_dur_s: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self { min_snr_db: MIN_SNR_DB, max_silence: MAX_SILENCE, min_dur_s: MIN_DUR_S }
    }
}

// ============================================================ 测量
pub struct Raw {
    pub container: &'static str,
    pub codec: &'static str,
    pub is_pcm: bool,
    pub bits: Option<u32>,
    pub channels: usize,
    pub side_ratio: Option<f64>,
    pub sr: u32,
    pub dur: f64,
    pub silence: f64,
    pub noise: Option<f64>,
    pub snr: Option<f64>,
    pub peak_db: Option<f64>,
    pub cutoff: Option<f64>,
    /// 低频端: 内容真正延伸到的最低 1/3 倍频程中心频率。
    pub low_edge: Option<f64>,
    pub flatness: Option<f64>,
    /// 20~40Hz 相对 100~300Hz 的落差 dB, 正数表示低频更弱。
    pub low_rolloff: Option<f64>,
    pub clip_events: usize,
    /// 掐头去尾之后仍然存在的绝对数字零总时长(秒)。
    pub zero_s: f64,
}

/// 容器格式看文件头, 不看后缀 —— 改个名字就叫 WAV 是这批素材最常见的问题。
fn container_of(path: &Path) -> &'static str {
    use std::io::Read;
    let mut head = [0u8; 12];
    let Ok(mut f) = std::fs::File::open(path) else { return "读不到" };
    if f.read_exact(&mut head).is_err() {
        return "读不到";
    }
    match (&head[0..4], &head[8..12]) {
        // RF64/BW64 是 WAV 的大文件扩展(>4GB), 一样是 RIFF 家族, 算 WAV
        (b"RIFF", b"WAVE") | (b"RF64", b"WAVE") | (b"BW64", b"WAVE") => "WAV",
        (b"fLaC", _) => "FLAC",
        (b"OggS", _) => "OGG",
        (b"FORM", b"AIFF") | (b"FORM", b"AIFC") => "AIFF",
        _ if &head[4..8] == b"ftyp" => "MP4/M4A",
        _ if head[0] == 0xFF && head[1] & 0xE0 == 0xE0 => "MP3",
        _ if &head[0..3] == b"ID3" => "MP3",
        _ => "未知",
    }
}

pub fn analyse(path: &Path, _cfg: &Settings) -> Result<Raw, String> {
    let container = container_of(path);
    let d = decode(path)?;
    let dur = d.mono.len() as f64 / d.sr as f64;
    let db = frame_db(&d.mono, d.sr);

    // 静音比例 = 1 - 活动比例, 用项目A 那套能量门限(本轨 P95-35dB 且高于 -55dBFS)。
    // 好处是跟素材录得多响无关; 代价是"静音"的定义是相对的, 不是一条绝对电平线。
    let silence = if dur > 0.0 { (1.0 - active_seconds(&db) / dur).clamp(0.0, 1.0) } else { 0.0 };

    let noise = noise_floor_db_above(&db, DIGITAL_ZERO_DBFS);
    // 信噪比 = 活动段 RMS - 底噪。两者都是能量平均, 直接相减就是 dB 差。
    let mask = energy_mask(&db);
    let signal = active_rms_db(&d.mono, d.sr, &mask);
    let snr = noise.map(|n| signal - n);

    let peak_db = (d.peak > 0.0).then(|| 20.0 * d.peak.log10());
    let cutoff = cutoff_hz(&d.mono, d.sr);
    let spec = ltas_db(&d.mono);
    let flatness = spec.as_ref().and_then(|s| flatness_db(s, d.sr));
    let low_rolloff = spec.as_ref().and_then(|s| low_rolloff_db(s, d.sr));
    let low_edge = spec.as_ref().and_then(|s| low_edge_hz(s, d.sr));

    Ok(Raw {
        container,
        codec: d.codec,
        is_pcm: d.is_pcm,
        bits: d.bits,
        channels: d.channels,
        side_ratio: d.side_ratio,
        sr: d.sr,
        dur,
        silence,
        noise,
        snr,
        peak_db,
        cutoff,
        low_edge,
        flatness,
        low_rolloff,
        clip_events: d.clip_events,
        zero_s: interior_zero_seconds(&db),
    })
}

/// 掐掉开头结尾的数字零(导出留白很常见), 统计中间还剩多少绝对零。
/// 中间出现真正的数字零, 说明这段被人为静音过 —— 话筒录出来的东西不可能是绝对零。
fn interior_zero_seconds(db: &[f64]) -> f64 {
    let is_zero = |v: &f64| *v <= DIGITAL_ZERO_DBFS;
    let lead = db.iter().take_while(|v| is_zero(v)).count();
    let tail = db.iter().rev().take_while(|v| is_zero(v)).count();
    if lead + tail >= db.len() {
        return 0.0;          // 整条全是零, 交给别的判定去说
    }
    db[lead..db.len() - tail].iter().filter(|v| is_zero(v)).count() as f64 * HOP
}

/// 1/3 倍频程带级 (中心频率, dB)。只取落在 20Hz~奈奎斯特之间、且至少覆盖 1 个
/// FFT bin 的频带 —— 8192 点 FFT 在 44.1k 下 bin 宽 5.4Hz, 最低那几个带本来就
/// 只盖得住一两个 bin, 数字不可信。
fn third_octave(spec: &[f64], sr: u32) -> Vec<(f64, f64)> {
    let bin_hz = sr as f64 / CUT_NFFT as f64;
    let nyq = sr as f64 / 2.0;
    let edge = 2f64.powf(1.0 / 6.0);        // 1/3 倍频程的上下边界系数
    let mut out = Vec::new();
    let mut fc = 25.0f64;                    // ISO 系列起点, 20Hz 那一带 bin 太少不要
    while fc <= BAND_HI_HZ {
        let (lo, hi) = (fc / edge, fc * edge);
        if hi <= nyq {
            let (a, b) = ((lo / bin_hz) as usize, (hi / bin_hz) as usize);
            let b = b.min(spec.len().saturating_sub(1));
            if b > a {
                // 带内按能量求和再转 dB, 不是把 dB 直接平均
                let p: f64 = spec[a..=b].iter().map(|v| 10f64.powf(v / 10.0)).sum();
                out.push((fc, 10.0 * (p / (b - a + 1) as f64).log10()));
            }
        }
        fc *= 2f64.powf(1.0 / 3.0);
    }
    out
}

/// 频响起伏 = 各 1/3 倍频程带级的 P95-P5 跨度。
///
/// 注意: 拿正常人声/音乐算出来的是**素材本身**的频谱起伏, 不是录音链路的频响。
/// 只有素材是扫频/粉噪这类已知激励时, 这个数才等于"频响 ±xx dB"。见"局限"。
fn flatness_db(spec: &[f64], sr: u32) -> Option<f64> {
    let bands = third_octave(spec, sr);
    if bands.len() < 8 {
        return None;
    }
    let mut lv: Vec<f64> = bands.iter().map(|(_, v)| *v).collect();
    lv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 用 P95-P5 而不是 max-min: 单个坏带(比如 50Hz 工频)不该把整个数字带跑
    Some(percentile(&lv, 95.0) - percentile(&lv, 5.0))
}

/// 内容真正延伸到的低频端: 从最低的 1/3 倍频程带往上找, 第一个"跟 100~300Hz
/// 参考段相差在 LOW_ROLLOFF_DB 以内"的带心频率。用来填「频响」列的下限 ——
/// 报 20Hz 得是真的有 20Hz 内容, 不能因为规格写着 20Hz 就照抄一个 20Hz 上去。
fn low_edge_hz(spec: &[f64], sr: u32) -> Option<f64> {
    let bands = third_octave(spec, sr);
    let refs: Vec<f64> =
        bands.iter().filter(|(f, _)| (100.0..=300.0).contains(f)).map(|(_, v)| *v).collect();
    if refs.is_empty() {
        return None;
    }
    let r = refs.iter().sum::<f64>() / refs.len() as f64;
    bands.iter().find(|(_, v)| r - *v <= LOW_ROLLOFF_DB).map(|(f, _)| *f)
}

/// 20~40Hz 相对 100~300Hz 的落差 dB。高通滤波会让这个数变得很大, 但人声素材
/// 本来低频就少, 所以只能当线索, 不能当判据。
fn low_rolloff_db(spec: &[f64], sr: u32) -> Option<f64> {
    let bands = third_octave(spec, sr);
    let avg = |lo: f64, hi: f64| -> Option<f64> {
        let v: Vec<f64> = bands.iter().filter(|(f, _)| *f >= lo && *f <= hi).map(|(_, v)| *v).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    Some(avg(100.0, 300.0)? - avg(BAND_LO_HZ, 40.0)?)
}

// ============================================================ 判定
/// 校验一个文件, 出一行。项目B 每个文件独立成行, 没有跨文件指标。
pub fn check_file(path: &Path, root: Option<&Path>, cfg: &Settings) -> Vec<String> {
    let name = match root {
        Some(r) => path.strip_prefix(r).unwrap_or(path).to_string_lossy().to_string(),
        None => path.file_name().unwrap_or_default().to_string_lossy().to_string(),
    };

    let r = match analyse(path, cfg) {
        Ok(r) => r,
        Err(e) => {
            let mut row = vec![String::new(); COLUMNS.len()];
            row[0] = name;
            row[1] = container_of(path).to_string();
            row[VERDICT_COL] = format!("否：{e}");
            return row;
        }
    };

    let mut fails: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let nyq = r.sr as f64 / 2.0;

    // 1. 必须是 WAV, 且里面装的是未压缩 PCM。
    //    .wav 容器照样能装 ADPCM / A-law / mp3, 光看容器判不出"未压缩"。
    if r.container != "WAV" {
        fails.push(format!("容器不是WAV(实测{})", r.container));
    }
    if !r.is_pcm {
        fails.push(format!("编码是{}, 不是未压缩PCM", r.codec));
    }

    // 2. 立体声
    if r.channels != REQ_CHANNELS {
        fails.push(format!("{}声道, 要求立体声", r.channels));
    }
    if r.side_ratio.is_some_and(|v| v < DUAL_MONO_SIDE) {
        notes.push("疑似双单声道(左右声道完全相同, 声道数够但没有立体声信息)".into());
    }

    // 3. 采样率
    if r.sr < MIN_SR {
        fails.push(format!("采样率{}Hz<44.1kHz", r.sr));
    }

    // 4. 时长与静音比例
    if r.dur < cfg.min_dur_s {
        fails.push(format!("时长{}不足{:.0}秒", mmss(r.dur), cfg.min_dur_s));
    }
    if r.silence > cfg.max_silence {
        fails.push(format!("静音比例{:.1}%>{:.0}%", r.silence * 100.0, cfg.max_silence * 100.0));
    }

    // 5a. 信噪比。测不出来只记备注不判不合格 —— 跟项目A 处理底噪/RT60 的口径一致:
    //     测不出不等于不合格, 但要让看表的人知道这项没验上。
    match r.snr {
        None => notes.push("信噪比未测出(没有够 0.5 秒的静音段可估底噪), 需人工复核".into()),
        Some(v) if v < cfg.min_snr_db => {
            fails.push(format!("信噪比{v:.1}dB<{:.0}dB", cfg.min_snr_db))
        }
        Some(_) => {}
    }

    // 5b. 频响上限 + 有损压缩痕迹。两件事共用同一个砖墙检测, 分开报:
    //     砖墙远低于奈奎斯特 = 这条轨过过有损编码(第6条);
    //     砖墙就在奈奎斯特附近但够不到 20kHz = 采样率不够(第5条)。
    match r.cutoff {
        None => notes.push("高频截止未测出".into()),
        Some(c) => {
            if c < nyq * LOSSY_NYQUIST_RATIO {
                fails.push(format!(
                    "高频在{:.1}kHz处砖墙跌落(奈奎斯特{:.1}kHz), 疑似有损压缩来源",
                    c / 1000.0,
                    nyq / 1000.0
                ));
            } else if c < BAND_HI_HZ {
                fails.push(format!("高频截止{:.1}kHz<20kHz", c / 1000.0));
            }
        }
    }
    if r.sr == 44100 {
        notes.push("44.1kHz 奈奎斯特只有 22.05kHz, 抗混叠滤波常在 20kHz 附近就开始滚降, 要稳过 20kHz 建议录 48kHz".into());
    }

    // 5c. 「频响」列里报的是实测带宽(低频端-高频截止), 参与判定的只有高频端。
    //     平坦度另报一个数进备注, 不参与判定, 原因见文件末尾"局限"。
    if let Some(f) = r.flatness {
        notes.push(format!(
            "频响起伏{f:.1}dB(1/3倍频程 P95-P5, 反映的是素材本身频谱, 不是链路频响; ±{FLATNESS_DB:.0}dB 需用扫频/粉噪测试信号才能判)"
        ));
    }
    // 信噪比不合格时把底噪一并写出来, 否则只看一个 SNR 数字没法判断是"底噪高"
    // 还是"录得太小声"。底噪本身没有单列一列。
    if let (Some(v), Some(n)) = (r.snr, r.noise) {
        if v < cfg.min_snr_db {
            notes.push(format!("底噪{n:.1}dBFS"));
        }
    }

    // 6. 后期处理痕迹。这几项都只是线索, 一律进备注不判不合格。
    if r.clip_events > 0 {
        notes.push(format!("检测到{}处削波", r.clip_events));
    }
    if r.peak_db.is_some_and(|p| p >= LIMITED_PEAK_DB) {
        notes.push("峰值贴满量程, 疑似做过限幅或归一化".into());
    }
    if r.low_rolloff.is_some_and(|v| v > LOW_ROLLOFF_DB) {
        notes.push(format!(
            "20-40Hz比100-300Hz低{:.0}dB, 可能做过高通滤波(也可能素材本身就没有低频内容)",
            r.low_rolloff.unwrap_or(0.0)
        ));
    }
    if r.zero_s > ZERO_RUN_S {
        notes.push(format!(
            "中段有{:.1}秒绝对数字零, 话筒录音不会出现, 疑似做过降噪门限或静音处理",
            r.zero_s
        ));
    }
    if r.bits.is_some_and(|b| b < 16) {
        notes.push(format!("位深仅{}bit", r.bits.unwrap_or(0)));
    }

    vec![
        name,
        fmt_file_type(&r),
        fmt_stereo(&r),
        r.channels.to_string(),
        r.sr.to_string(),
        mmss(r.dur),
        format!("{:.1}%", r.silence * 100.0),
        fmt_opt(r.snr, 1),
        fmt_band(&r),
        if fails.is_empty() { "是".into() } else { format!("否：{}", fails.join("；")) },
        notes.join("；"),
    ]
}

/// 「文件类型」一列要同时回答第 1 条(必须是 WAV)和第 6 条(未压缩), 所以容器和
/// 编码都写进去: "WAV/pcm_s16le 16bit"。位深附在后面, 不单独占一列。
fn fmt_file_type(r: &Raw) -> String {
    let mut t = format!("{}/{}", r.container, r.codec);
    if let Some(b) = r.bits {
        t.push_str(&format!(" {b}bit"));
    }
    t
}

/// 「是否立体声」。双单声道声道数确实是 2, 判定上不算不合格(见备注), 但这一列
/// 必须把它跟真立体声区分开 —— 否则一条左右完全相同的轨在表里跟合格的长得一样。
fn fmt_stereo(r: &Raw) -> String {
    if r.channels != REQ_CHANNELS {
        return "否".into();
    }
    match r.side_ratio {
        Some(v) if v < DUAL_MONO_SIDE => "是(左右相同)".into(),
        _ => "是".into(),
    }
}

/// 「频响」一列报实测带宽。低频端是内容真正延伸到的最低 1/3 倍频程带, 高频端是
/// 砖墙检测的截止点 —— 不是照抄规格里的 "20Hz-20kHz"。测不出就留空。
fn fmt_band(r: &Raw) -> String {
    match (r.low_edge, r.cutoff) {
        (Some(lo), Some(hi)) => format!("{lo:.0}Hz-{:.1}kHz", hi / 1000.0),
        (None, Some(hi)) => format!("?-{:.1}kHz", hi / 1000.0),
        _ => String::new(),
    }
}

fn fmt_opt(v: Option<f64>, n: usize) -> String {
    v.map(|x| format!("{x:.*}", n)).unwrap_or_default()
}

fn mmss(s: f64) -> String {
    let t = s.round().max(0.0) as u64;
    format!("{}:{:02}", t / 60, t % 60)
}

// ============================================================ 局限
//
// **频响 20Hz-20kHz ±3dB 测不出来。** 频响是录音**链路**的属性, 只有用已知激励
// (扫频/粉噪)去激励、再看输出偏离多少才叫频响。拿一段人声或音乐算长时平均谱,
// 得到的是内容本身的频谱 —— 任何真实素材都是低频重、高频滚降, 离"±3dB 平坦"
// 差三四十 dB, 判出来会全军覆没, 而且这个"不合格"不说明任何问题。所以本工具
// 把这条拆成两半: **上限**用砖墙检测判(高频有没有被截掉, 见 5b), **平坦度**只
// 在备注里报个参考数, 不参与合格判定。要真校验 ±3dB, 得让交付方每批附一段扫频
// 或粉噪校准文件, 那时候 `flatness_db` 算出来的数才有物理意义。
//
// **"未经算法处理"只能查到硬证据。** 判不合格的只有两样: 编码不是 PCM(第6条
// 直接违反), 以及高频砖墙远低于奈奎斯特(过过有损编码)。降噪、限幅、高通滤波
// 这些在波形上只留统计线索, 反过来推一定会误报 —— 一条录得很干净、本来低频
// 就少的人声, 跟一条做过高通的, 频谱上可以长得一模一样。所以这几项一律写进
// 备注交人工复核, 不替人做"退货"的决定。

#[cfg(test)]
mod tests {
    use super::*;

    /// 写一个 16bit PCM WAV。测试要走完真实的解码路径, 光靠合成的 f64 数组
    /// 验不到容器识别、编码识别、位深、声道这几项。
    fn write_wav(path: &Path, sr: u32, ch: u16, frames: &[Vec<f64>]) {
        use std::io::Write;
        let n = frames.len() * ch as usize;
        let data_len = n * 2;
        let mut b: Vec<u8> = Vec::with_capacity(44 + data_len);
        b.extend(b"RIFF");
        b.extend(((36 + data_len) as u32).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend(1u16.to_le_bytes());                       // PCM
        b.extend(ch.to_le_bytes());
        b.extend(sr.to_le_bytes());
        b.extend((sr * ch as u32 * 2).to_le_bytes());        // 字节率
        b.extend((ch * 2).to_le_bytes());                    // 块对齐
        b.extend(16u16.to_le_bytes());                       // 位深
        b.extend(b"data");
        b.extend((data_len as u32).to_le_bytes());
        for fr in frames {
            for v in fr {
                b.extend(((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
            }
        }
        std::fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    /// 3.3 秒有内容 + 0.7 秒仅本底噪声的立体声轨。左右给不同相位, 是真立体声。
    ///
    /// 本底幅度必须大于 16bit 的 1 个 LSB(1/32767 = 3.05e-5), 否则量化之后整段
    /// 变成绝对数字零, 会被 DIGITAL_ZERO_DBFS 当数字静音滤掉、底噪测不出来 ——
    /// 这正是本测试第一次写错的地方。取 1.5 LSB, 量化后本底约 -91dBFS,
    /// 相对 -13.5dBFS 的信号是 ~78dB 信噪比, 稳稳过 70dB。
    /// 静音段 0.7 秒也是刻意的: 要过 analysis::MIN_SILENCE_S(0.5 秒)这道门槛,
    /// 而 0.7/4.0 = 17.5% 又还在 20% 的静音比例上限之内。
    fn stereo_take(sr: u32, dual_mono: bool) -> Vec<Vec<f64>> {
        const DITHER: f64 = 1.5 / 32767.0;
        let mut out = Vec::new();
        let total = (sr as f64 * 4.0) as usize;
        let loud = (sr as f64 * 3.3) as usize;
        for i in 0..total {
            let t = i as f64 / sr as f64;
            // 伪随机本底: 不用 rand 依赖, 线性同余够了
            let noise = |k: u64| {
                let h = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((h >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * DITHER
            };
            let (mut l, mut r) = (noise(i as u64), noise(i as u64 + 7_777_777));
            if i < loud {
                let w = 2.0 * std::f64::consts::PI * 440.0 * t;
                l += 0.3 * w.sin();
                r += 0.3 * if dual_mono { w.sin() } else { (w + 1.2).sin() };
            }
            out.push(vec![l, r]);
        }
        out
    }

    struct Tmp(std::path::PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn tmp(tag: &str) -> Tmp {
        Tmp(std::env::temp_dir().join(format!("qc_raw_test_{tag}_{}.wav", std::process::id())))
    }

    /// 一条各项都达标的立体声 WAV 应该判"是"。时长下限放宽到 1 秒 ——
    /// 测试不值得为了凑 1 分钟去写 10MB 的文件。
    #[test]
    fn clean_stereo_wav_passes() {
        let f = tmp("ok");
        write_wav(&f.0, 44100, 2, &stereo_take(44100, false));
        let cfg = Settings { min_dur_s: 1.0, ..Default::default() };
        let r = analyse(&f.0, &cfg).expect("应该能解码");
        assert_eq!(r.container, "WAV");
        assert!(r.is_pcm, "编码是 {}", r.codec);
        assert_eq!(r.channels, 2);
        assert_eq!(r.sr, 44100);
        assert_eq!(r.bits, Some(16));
        assert!(r.snr.expect("应该测出信噪比") > MIN_SNR_DB, "snr={:?}", r.snr);
        // 纯音+白噪没有砖墙, 截止频率应该是满带宽
        assert!(r.cutoff.unwrap() > BAND_HI_HZ, "cutoff={:?}", r.cutoff);

        let row = check_file(&f.0, None, &cfg);
        assert_eq!(row[VERDICT_COL], "是", "备注: {}", row[NOTES_COL]);
        // 交付方指定的那几列要填对
        let col = |n: &str| row[COLUMNS.iter().position(|c| *c == n).unwrap()].clone();
        assert!(col("文件类型").starts_with("WAV/"), "{}", col("文件类型"));
        assert!(col("文件类型").ends_with("16bit"), "{}", col("文件类型"));
        assert_eq!(col("是否立体声"), "是");
        assert_eq!(col("声道数"), "2");
        assert_eq!(col("采样率"), "44100");
        assert_eq!(col("时长"), "0:04");
        assert!(col("静音比例").ends_with('%'), "{}", col("静音比例"));
        assert!(col("频响").ends_with("kHz"), "{}", col("频响"));
    }

    /// 默认时长下限是 1 分钟, 3 秒的文件必须因此不合格。
    #[test]
    fn short_take_fails_duration() {
        let f = tmp("short");
        write_wav(&f.0, 44100, 2, &stereo_take(44100, false));
        let row = check_file(&f.0, None, &Settings::default());
        assert!(row[VERDICT_COL].contains("时长"), "{}", row[VERDICT_COL]);
    }

    /// 左右完全相同: 声道数是 2, 该判合格, 但要在备注里点出没有立体声信息。
    #[test]
    fn dual_mono_is_flagged_in_notes() {
        let f = tmp("dual");
        write_wav(&f.0, 44100, 2, &stereo_take(44100, true));
        let cfg = Settings { min_dur_s: 1.0, ..Default::default() };
        let row = check_file(&f.0, None, &cfg);
        assert!(row[NOTES_COL].contains("双单声道"), "备注: {}", row[NOTES_COL]);
        // 声道数还是 2, 但"是否立体声"这一列必须把它跟真立体声区分开
        assert_eq!(row[COLUMNS.iter().position(|c| *c == "是否立体声").unwrap()], "是(左右相同)");
        assert_eq!(row[COLUMNS.iter().position(|c| *c == "声道数").unwrap()], "2");
    }

    /// 单声道要判不合格(第 2 条要求立体声), 且 side_ratio 应为 None。
    #[test]
    fn mono_fails_channel_check() {
        let f = tmp("mono");
        let frames: Vec<Vec<f64>> =
            stereo_take(44100, false).into_iter().map(|fr| vec![fr[0]]).collect();
        write_wav(&f.0, 44100, 1, &frames);
        let cfg = Settings { min_dur_s: 1.0, ..Default::default() };
        let r = analyse(&f.0, &cfg).unwrap();
        assert_eq!(r.side_ratio, None);
        let row = check_file(&f.0, None, &cfg);
        assert!(row[VERDICT_COL].contains("要求立体声"), "{}", row[VERDICT_COL]);
        assert_eq!(row[COLUMNS.iter().position(|c| *c == "是否立体声").unwrap()], "否");
    }

    /// 22.05kHz 采样: 采样率不合格, 且奈奎斯特只有 11kHz, 高频也够不到 20kHz。
    #[test]
    fn low_sample_rate_fails() {
        let f = tmp("lowsr");
        write_wav(&f.0, 22050, 2, &stereo_take(22050, false));
        let cfg = Settings { min_dur_s: 1.0, ..Default::default() };
        let row = check_file(&f.0, None, &cfg);
        assert!(row[VERDICT_COL].contains("采样率"), "{}", row[VERDICT_COL]);
        assert!(row[VERDICT_COL].contains("20kHz"), "{}", row[VERDICT_COL]);
    }

    #[test]
    fn columns_and_verdict_position() {
        assert_eq!(COLUMNS[VERDICT_COL], "是否满足要求");
        assert_eq!(COLUMNS[NOTES_COL], "备注（如有）");
    }

    /// 交付方指定的九个字段一个都不能少、顺序也不能乱。
    #[test]
    fn delivery_required_columns_present_in_order() {
        let want = [
            "文件类型", "是否立体声", "声道数", "采样率", "时长",
            "静音比例", "信噪比dB", "频响", "是否满足要求",
        ];
        let got: Vec<&str> =
            COLUMNS.iter().copied().filter(|c| want.contains(c)).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn third_octave_stops_at_nyquist() {
        let spec = vec![-30.0f64; CUT_NFFT / 2 + 1];
        let bands = third_octave(&spec, 44100);
        assert!(!bands.is_empty());
        assert!(bands.iter().all(|(f, _)| *f * 2f64.powf(1.0 / 6.0) <= 22050.0));
        // 平谱的起伏应该接近 0
        assert!(flatness_db(&spec, 44100).unwrap() < 0.5);
    }

    #[test]
    fn interior_zero_ignores_head_and_tail() {
        let z = -200.0;
        // 头 3 帧 + 尾 2 帧是导出留白, 不该计入; 中间 4 帧才算
        let db = [z, z, z, -40.0, z, z, z, z, -40.0, z, z];
        let s = interior_zero_seconds(&db);
        assert!((s - 4.0 * HOP).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn all_zero_track_reports_none() {
        assert_eq!(interior_zero_seconds(&[-200.0; 10]), 0.0);
    }
}
