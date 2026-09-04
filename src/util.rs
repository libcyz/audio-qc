//! 界面外壳里跟音频无关的零件。两个校验项目的窗口长得不一样, 但字体、输出路径、
//! 写 CSV、用系统程序打开文件这几件事完全一致, 放这儿共用。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;

/// egui 自带字体不含中文, 不装的话全是豆腐块。直接借系统字体, 不往可执行文件里塞 10MB。
pub fn install_cjk_font(ctx: &egui::Context) {
    // 一个中文字体都没找到也不崩, 只是显示成豆腐块
    let Some((bytes, index)) = cjk_font() else { return };
    let mut fonts = egui::FontDefinitions::default();
    let mut data = egui::FontData::from_owned(bytes);
    data.index = index;
    fonts.font_data.insert("cjk".to_owned(), Arc::new(data));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().insert(0, "cjk".to_owned());
    }
    ctx.set_fonts(fonts);
}

/// 读出第一个能用的中文字体, 返回 (字体文件内容, ttc 里的字体下标)。
fn cjk_font() -> Option<(Vec<u8>, u32)> {
    candidates()
        .into_iter()
        .find_map(|(path, index)| std::fs::read(&path).ok().map(|b| (b, index)))
}

#[cfg(target_os = "windows")]
fn candidates() -> Vec<(PathBuf, u32)> {
    [
        ("C:/Windows/Fonts/msyh.ttc", 0),      // 微软雅黑
        ("C:/Windows/Fonts/msyh.ttf", 0),
        ("C:/Windows/Fonts/simhei.ttf", 0),    // 黑体
        ("C:/Windows/Fonts/simsun.ttc", 0),    // 宋体
        ("C:/Windows/Fonts/msjh.ttc", 0),      // 微軟正黑 (繁体系统)
    ]
    .into_iter()
    .map(|(p, i)| (PathBuf::from(p), i))
    .collect()
}

#[cfg(target_os = "macos")]
fn candidates() -> Vec<(PathBuf, u32)> {
    [
        ("/System/Library/Fonts/PingFang.ttc", 0),          // 苹方
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0),    // 黑体-简
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/Library/Fonts/Arial Unicode.ttf", 0),
    ]
    .into_iter()
    .map(|(p, i)| (PathBuf::from(p), i))
    .collect()
}

/// Linux 的字体路径各发行版完全对不上 —— 同一个 Noto Sans CJK, Debian/Ubuntu 在
/// `/usr/share/fonts/opentype/noto/`, Fedora 在 `/usr/share/fonts/google-noto-sans-cjk-vf-fonts/`,
/// Arch 在 `/usr/share/fonts/noto-cjk/`。原来写死三条 Debian 路径, 换个发行版就一条
/// 都命中不了, 整个界面变成豆腐块(这就是 Linux 版"乱码"的成因)。
/// 改成先问 fontconfig(它按字符覆盖率答, 最准), 问不到再自己扫字体目录。
///
/// ttc 下标一律取 0: 中文 ttc 里每个 face 都是同一套汉字字形, 差别只在日韩异体和
/// 字重; 而 fontconfig 给变体字体报的下标是 (实例号<<16|face号), 塞给 egui 反而越界。
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn candidates() -> Vec<(PathBuf, u32)> {
    let mut found = fontconfig_zh();
    if found.is_empty() {
        found = scan_font_dirs();
    }
    // 同一个文件会被 fontconfig 按 face 报很多遍, 排序时顺手去重
    found.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.cmp(b)));
    found.dedup();
    found.into_iter().map(|p| (p, 0)).collect()
}

/// 问 fontconfig 要"真正覆盖简体中文"的字体文件。fc-list 只列覆盖得了的,
/// 所以列表空就是系统里真没有中文字体(fc-match 做不到这点, 它总会答一个)。
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn fontconfig_zh() -> Vec<PathBuf> {
    let Ok(out) = std::process::Command::new("fc-list")
        .args([":lang=zh-cn", "--format=%{file}\n"])
        .output()
    else {
        return Vec::new();   // 没装 fontconfig
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// 没有 fontconfig 时的兜底: 扫字体目录, 按文件名认中文字体。
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn scan_font_dirs() -> Vec<PathBuf> {
    // 只留一眼能认出是中文字体的名字, 宁可漏也别把西文字体挑进来 —— 挑错了
    // 界面照样是豆腐块, 还更难查。
    const NAMES: [&str; 10] = [
        "cjk", "droidsansfallback", "wqy", "sourcehans", "notosanssc", "notosanstc",
        "notoserifsc", "sarasa", "uming", "ukai",
    ];
    const EXTS: [&str; 4] = ["ttc", "ttf", "otf", "otc"];

    let mut dirs = vec![
        PathBuf::from("/usr/share/fonts"),
        PathBuf::from("/usr/local/share/fonts"),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(&home).join(".local/share/fonts"));
        dirs.push(PathBuf::from(&home).join(".fonts"));
    }

    let mut out = Vec::new();
    for dir in dirs {
        for e in walkdir::WalkDir::new(dir).max_depth(4).into_iter().filter_map(Result::ok) {
            let p = e.path();
            let ok_ext = p
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| EXTS.contains(&x.to_ascii_lowercase().as_str()));
            if !ok_ext {
                continue;
            }
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
            if NAMES.iter().any(|k| name.contains(k)) {
                out.push(p.to_path_buf());
            }
        }
    }
    out
}

/// 排序用的偏好: 数越小越先用。都能显示中文, 挑最适合表格正文的那个。
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn rank(p: &Path) -> i32 {
    let n = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
    let mut r = 0;
    if n.contains("serif") || n.contains("ming") || n.contains("song") || n.contains("kai") {
        r += 20;    // 衬线/宋体/楷体: 小字号下不如黑体清楚
    }
    if n.contains("mono") {
        r += 10;    // 等宽 CJK 的汉字被压窄, 表格里更难认
    }
    if n.contains("fallback") || n.contains("unifont") {
        r += 5;     // 只为"别缺字"存在的兜底字体, 字形质量一般
    }
    if n.contains("cjk") || n.contains("sourcehans") || n.contains("hei") {
        r -= 10;    // Noto Sans CJK / 思源黑体 / 各种黑体: 表格正文最合适
    }
    r
}

/// 结果写在拖进来的东西旁边; 只读目录(比如网络盘)就退回桌面。
pub fn out_path(first: &Path) -> PathBuf {
    let base = if first.is_dir() {
        first.to_path_buf()
    } else {
        first.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let name = format!("校验结果_{}.csv", timestamp());
    // 试着在目标目录建个临时文件, 建不了就说明没写权限
    let probe = base.join(format!(".写权限测试_{}", std::process::id()));
    match std::fs::write(&probe, b"") {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            base.join(name)
        }
        Err(_) => desktop().join(name),
    }
}

pub fn desktop() -> PathBuf {
    let home = if cfg!(target_os = "windows") { "USERPROFILE" } else { "HOME" };
    std::env::var_os(home)
        .map(|h| PathBuf::from(h).join("Desktop"))
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
}

/// 用系统默认程序打开文件。
pub fn open_in_system(path: &Path) {
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]).arg(path);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = std::process::Command::new("open");
        c.arg(path);
        c
    };
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let mut cmd = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(path);
        c
    };
    let _ = cmd.spawn();
}

pub fn timestamp() -> String {
    // 只为了给文件名去重, 不值得为此引入 chrono
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86400;
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let (y, mo, d) = civil_from_days(days as i64);
    format!("{y:04}{mo:02}{d:02}_{h:02}{m:02}{s:02}")
}

/// days since 1970-01-01 -> (年, 月, 日)。Howard Hinnant 的历法算法。
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 写 csv。防呆: 文件被 Excel 占着就自动换个名字, 别让一次校验白跑。
pub fn write_csv(out: &Path, columns: &[&str], rows: &[Vec<String>]) -> Result<PathBuf, String> {
    let mut target = out.to_path_buf();
    for attempt in 0..20 {
        if attempt > 0 {
            let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("校验结果");
            target = out.with_file_name(format!("{stem}({}).csv", attempt + 1));
        }
        match try_write(&target, columns, rows) {
            Ok(()) => return Ok(target),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => continue,
            Err(e) => return Err(format!("写不进去: {e}")),
        }
    }
    Err("结果文件一直被占用(是不是在 Excel 里开着?)".into())
}

fn try_write(target: &Path, columns: &[&str], rows: &[Vec<String>]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(target)?;
    f.write_all(b"\xEF\xBB\xBF")?;      // BOM: Excel 打开中文才不乱码
    let mut w = csv::Writer::from_writer(f);
    w.write_record(columns).map_err(std::io::Error::other)?;
    for r in rows {
        w.write_record(r).map_err(std::io::Error::other)?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 挑出来的字体必须真的画得出汉字 —— 光"文件读得进来"不算数, Linux 版之前
    /// 就是路径写死后一个都没命中, 整个界面变豆腐块。这里把字体装进一个离屏
    /// Context 跑一帧, 直接问 egui 认不认得这几个字。
    #[test]
    fn picked_font_can_draw_chinese() {
        let Some(_) = cjk_font() else {
            eprintln!("本机没装中文字体, 跳过(裸容器里正常)");
            return;
        };
        let ctx = egui::Context::default();
        install_cjk_font(&ctx);
        let mut out = ctx.run_ui(Default::default(), |_| {});
        // 没有渲染后端接手这帧生成的字形贴图, 不清掉的话 FullOutput 析构时会 panic
        out.textures_delta.clear();
        let ok = ctx.fonts_mut(|f| {
            f.has_glyphs(&egui::FontId::proportional(14.0), "文件名备注截止频率")
        });
        assert!(ok, "选中的字体画不出汉字");
    }
}
