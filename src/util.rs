//! 界面外壳里跟音频无关的零件。两个校验项目的窗口长得不一样, 但字体、输出路径、
//! 写 CSV、用系统程序打开文件这几件事完全一致, 放这儿共用。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;

/// egui 自带字体不含中文, 不装的话全是豆腐块。直接借系统字体, 不往可执行文件里塞 10MB。
pub fn install_cjk_font(ctx: &egui::Context) {
    #[cfg(target_os = "windows")]
    const CANDIDATES: [(&str, u32); 5] = [
        ("C:/Windows/Fonts/msyh.ttc", 0),      // 微软雅黑
        ("C:/Windows/Fonts/msyh.ttf", 0),
        ("C:/Windows/Fonts/simhei.ttf", 0),    // 黑体
        ("C:/Windows/Fonts/simsun.ttc", 0),    // 宋体
        ("C:/Windows/Fonts/msjh.ttc", 0),      // 微軟正黑 (繁体系统)
    ];
    #[cfg(target_os = "macos")]
    const CANDIDATES: [(&str, u32); 4] = [
        ("/System/Library/Fonts/PingFang.ttc", 0),          // 苹方
        ("/System/Library/Fonts/STHeiti Medium.ttc", 0),    // 黑体-简
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
        ("/Library/Fonts/Arial Unicode.ttf", 0),
    ];
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    const CANDIDATES: [(&str, u32); 3] = [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
        ("/usr/share/fonts/truetype/arphic/uming.ttc", 0),
    ];
    for (path, index) in CANDIDATES {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let mut fonts = egui::FontDefinitions::default();
        let mut data = egui::FontData::from_owned(bytes);
        data.index = index;
        fonts.font_data.insert("cjk".to_owned(), Arc::new(data));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().insert(0, "cjk".to_owned());
        }
        ctx.set_fonts(fonts);
        return;
    }
    // 一个中文字体都没有也不崩, 只是显示成豆腐块
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
