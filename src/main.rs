#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]
//! 音频实录质量校验 - 拖拽版 (Rust)
//!
//! 把音频文件或文件夹拖进窗口即可。只校验音频本身, 不看 meta json。

mod analysis;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use analysis::{check_song, group_audio, SongGroup, AUDIO_EXT, COLUMNS};
use eframe::egui;

const MAX_SCAN_FILES: usize = 5000;   // 防呆: 拖进来一整个盘时别无限扫下去

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 660.0])
            .with_min_inner_size([760.0, 480.0])
            .with_drag_and_drop(true)
            .with_title("音频实录质量校验"),
        ..Default::default()
    };
    eframe::run_native(
        "音频实录质量校验",
        options,
        Box::new(|cc| {
            install_cjk_font(&cc.egui_ctx);
            let mut app = App::default();
            // 支持把文件/文件夹直接拖到 exe 图标上
            let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
            if !args.is_empty() {
                app.start(args);
            }
            Ok(Box::new(app))
        }),
    )
}

/// egui 自带字体不含中文, 不装的话全是豆腐块。直接借系统字体, 不往可执行文件里塞 10MB。
fn install_cjk_font(ctx: &egui::Context) {
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

enum Msg {
    Status(String),
    Rows(Vec<Vec<String>>, usize),
    Done { out: Option<PathBuf>, songs: usize, files: usize, bad: usize, err: Option<String> },
}

#[derive(Default)]
struct App {
    rows: Vec<Vec<String>>,
    status: String,
    hint: String,
    progress: (usize, usize),
    busy: bool,
    csv: Option<PathBuf>,
    rx: Option<Receiver<Msg>>,
    cancel: Option<Arc<AtomicBool>>,
    toast: String,
}

impl App {
    fn start(&mut self, paths: Vec<PathBuf>) {
        if self.busy {
            self.toast = "正在校验中，等这批跑完再拖".into();
            return;
        }
        // 防呆: 去重 + 丢掉不存在的路径
        let mut seen = std::collections::HashSet::new();
        let paths: Vec<PathBuf> = paths
            .into_iter()
            .filter_map(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
            .filter(|p| seen.insert(p.clone()))
            .collect();
        let missing: Vec<&PathBuf> = paths.iter().filter(|p| !p.exists()).collect();
        if !missing.is_empty() {
            self.toast = format!("有 {} 个路径不存在，已跳过", missing.len());
        }
        let paths: Vec<PathBuf> = paths.into_iter().filter(|p| p.exists()).collect();
        if paths.is_empty() {
            self.toast = "没有可校验的路径".into();
            return;
        }

        let (jobs, truncated) = collect(&paths);
        if jobs.is_empty() {
            self.toast = "拖进来的位置里没找到音频文件".into();
            return;
        }
        if truncated {
            self.toast = format!("文件太多，只取前 {MAX_SCAN_FILES} 个音频");
        }

        let out = out_path(&paths[0]);
        let total = jobs.len();
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));

        self.rows.clear();
        self.csv = None;
        self.busy = true;
        self.progress = (0, total);
        self.status = "开始校验…".into();
        self.rx = Some(rx);
        self.cancel = Some(cancel.clone());

        let root = if paths.len() == 1 && paths[0].is_dir() {
            Some(paths[0].clone())
        } else {
            None
        };
        std::thread::spawn(move || worker(jobs, root, out, tx, cancel));
    }
}

fn worker(
    jobs: Vec<SongGroup>,
    root: Option<PathBuf>,
    out: PathBuf,
    tx: Sender<Msg>,
    cancel: Arc<AtomicBool>,
) {
    let total = jobs.len();
    let mut all: Vec<Vec<String>> = Vec::new();
    let mut bad_songs = 0usize;

    for (i, g) in jobs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let _ = tx.send(Msg::Status(format!("({}/{}) {}", i + 1, total, g.title)));
        let rows = check_song(g, root.as_deref());
        if rows.iter().any(|r| r[10] != "是") {
            bad_songs += 1;
        }
        all.extend(rows.clone());
        let _ = tx.send(Msg::Rows(rows, i + 1));
    }

    let (out, err) = match write_csv(&out, &all) {
        Ok(p) => (Some(p), None),
        Err(e) => (None, Some(e)),
    };
    let _ = tx.send(Msg::Done {
        out,
        songs: total,
        files: all.len(),
        bad: bad_songs,
        err,
    });
}

/// 递归收集音频并按歌名分组。
fn collect(paths: &[PathBuf]) -> (Vec<SongGroup>, bool) {
    let mut jobs = Vec::new();
    let mut loose: Vec<PathBuf> = Vec::new();
    let mut count = 0usize;
    let mut truncated = false;

    let is_audio = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| AUDIO_EXT.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    };

    for p in paths {
        if p.is_dir() {
            // 每个目录单独分组: 同名不同目录的文件不该被算成一首歌
            let mut by_dir: std::collections::BTreeMap<PathBuf, Vec<PathBuf>> = Default::default();
            for entry in walkdir::WalkDir::new(p).follow_links(false).into_iter().flatten() {
                if count >= MAX_SCAN_FILES {
                    truncated = true;
                    break;
                }
                let f = entry.path();
                if f.is_file() && is_audio(f) {
                    by_dir.entry(f.parent().unwrap_or(p).to_path_buf())
                        .or_default()
                        .push(f.to_path_buf());
                    count += 1;
                }
            }
            for (_, mut files) in by_dir {
                files.sort();
                jobs.extend(group_audio(&files));
            }
        } else if is_audio(p) {
            loose.push(p.clone());
            count += 1;
        }
    }
    if !loose.is_empty() {
        loose.sort();
        jobs.extend(group_audio(&loose));
    }
    (jobs, truncated)
}

/// 结果写在拖进来的东西旁边; 只读目录(比如网络盘)就退回桌面。
fn out_path(first: &Path) -> PathBuf {
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

fn desktop() -> PathBuf {
    let home = if cfg!(target_os = "windows") { "USERPROFILE" } else { "HOME" };
    std::env::var_os(home)
        .map(|h| PathBuf::from(h).join("Desktop"))
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
}

/// 用系统默认程序打开文件。
fn open_in_system(path: &Path) {
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

fn timestamp() -> String {
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
fn write_csv(out: &Path, rows: &[Vec<String>]) -> Result<PathBuf, String> {
    let mut target = out.to_path_buf();
    for attempt in 0..20 {
        if attempt > 0 {
            let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("校验结果");
            target = out.with_file_name(format!("{stem}({}).csv", attempt + 1));
        }
        match try_write(&target, rows) {
            Ok(()) => return Ok(target),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => continue,
            Err(e) => return Err(format!("写不进去: {e}")),
        }
    }
    Err("结果文件一直被占用(是不是在 Excel 里开着?)".into())
}

fn try_write(target: &Path, rows: &[Vec<String>]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(target)?;
    f.write_all(b"\xEF\xBB\xBF")?;      // BOM: Excel 打开中文才不乱码
    let mut w = csv::Writer::from_writer(f);
    w.write_record(COLUMNS).map_err(std::io::Error::other)?;
    for r in rows {
        w.write_record(r).map_err(std::io::Error::other)?;
    }
    w.flush()
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let ctx = &ctx;
        // 拖进来的文件
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect()
        });
        if !dropped.is_empty() {
            self.start(dropped);
        }
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());

        // 后台消息
        let mut finished = false;
        if let Some(rx) = &self.rx {
            loop {
                let msg = match rx.try_recv() {
                    Ok(m) => m,
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    // 防呆: 工作线程万一 panic 了, 别让界面永远卡在"正在校验"
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        if self.busy {
                            self.busy = false;
                            self.status = "校验意外中断了(可以重试, 或换个文件看看)".into();
                            finished = true;
                        }
                        break;
                    }
                };
                match msg {
                    Msg::Status(s) => self.status = s,
                    Msg::Rows(rows, done) => {
                        self.rows.extend(rows);
                        self.progress.0 = done;
                    }
                    Msg::Done { out, songs, files, bad, err } => {
                        self.busy = false;
                        self.csv = out.clone();
                        self.hint = format!("{songs} 首 / {files} 个音频，不合格 {bad} 首");
                        self.status = match (err, out) {
                            (Some(e), _) => format!("结果没保存成: {e}"),
                            (None, Some(p)) => format!("结果已保存: {}", p.display()),
                            (None, None) => "完成".into(),
                        };
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.rx = None;
            self.cancel = None;
        }
        if self.busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::Panel::top("head").show(ui, |ui| {
            ui.add_space(8.0);
            let drop_text = if self.busy {
                "正在校验，请稍候…"
            } else if hovering {
                "松手开始校验"
            } else if self.hint.is_empty() {
                "把音频文件或文件夹拖到这里"
            } else {
                &self.hint
            };
            let fill = if hovering {
                ui.visuals().selection.bg_fill.gamma_multiply(0.35)
            } else {
                ui.visuals().faint_bg_color
            };
            egui::Frame::default()
                .fill(fill)
                .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(12, 22))
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new(drop_text).size(17.0));
                        if !self.hint.is_empty() && !self.busy {
                            ui.label(egui::RichText::new("可以继续拖下一批").weak());
                        }
                    });
                    ui.set_min_width(ui.available_width());
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.busy, egui::Button::new("选择文件…")).clicked() {
                    if let Some(f) = rfd::FileDialog::new()
                        .add_filter("音频", &AUDIO_EXT)
                        .pick_files()
                    {
                        self.start(f);
                    }
                }
                if ui.add_enabled(!self.busy, egui::Button::new("选择文件夹…")).clicked() {
                    if let Some(d) = rfd::FileDialog::new().pick_folder() {
                        self.start(vec![d]);
                    }
                }
                let has_csv = self.csv.is_some();
                if ui.add_enabled(has_csv, egui::Button::new("打开结果CSV")).clicked() {
                    if let Some(p) = &self.csv {
                        open_in_system(p);
                    }
                }
                if self.busy && ui.button("停止").clicked() {
                    if let Some(c) = &self.cancel {
                        c.store(true, Ordering::Relaxed);
                    }
                    self.status = "正在停止…".into();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(&self.status).weak());
                });
            });

            let (done, total) = self.progress;
            if total > 0 {
                ui.add_space(4.0);
                ui.add(
                    egui::ProgressBar::new(done as f32 / total as f32)
                        .text(format!("{done}/{total}"))
                        .desired_height(8.0),
                );
            }
            ui.add_space(8.0);
        });

        if !self.toast.is_empty() {
            let mut open = true;
            egui::Window::new("提示")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(&self.toast);
                    if ui.button("知道了").clicked() {
                        self.toast.clear();
                    }
                });
            if !open {
                self.toast.clear();
            }
        }

        egui::CentralPanel::default().show(ui, |ui| {
            result_table::table(ui, &self.rows);
        });
    }
}

/// 结果表格。没引 egui_extras, 手写一个够用的定宽表。
mod result_table {
    use eframe::egui;

    const HEADS: [(&str, f32); 6] = [
        ("文件名", 260.0),
        ("格式", 46.0),
        ("时长", 56.0),
        ("码率", 60.0),
        ("底噪", 62.0),
        ("截止", 58.0),
    ];

    pub fn table(ui: &mut egui::Ui, rows: &[Vec<String>]) {
        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("result")
                .striped(true)
                .spacing([10.0, 6.0])
                .show(ui, |ui| {
                    for (h, w) in HEADS {
                        ui.add_sized([w, 18.0], egui::Label::new(egui::RichText::new(h).strong()).truncate());
                    }
                    ui.add_sized([46.0, 18.0], egui::Label::new(egui::RichText::new("结论").strong()));
                    ui.label(egui::RichText::new("原因 / 备注").strong());
                    ui.end_row();

                    for r in rows {
                        // 列顺序: 0文件名 1格式 2采样率 3时长 4码率 5底噪 6截止 ... 10结论 11备注
                        for (i, (_, w)) in [0usize, 1, 3, 4, 5, 6].iter().zip(HEADS) {
                            ui.add_sized(
                                [w, 18.0],
                                egui::Label::new(r.get(*i).cloned().unwrap_or_default()).truncate(),
                            )
                            .on_hover_text(r.get(*i).cloned().unwrap_or_default());
                        }
                        let ok = r.get(10).map(|v| v == "是").unwrap_or(false);
                        let (verdict, color) = if ok {
                            ("是", egui::Color32::from_rgb(26, 127, 55))
                        } else {
                            ("否", egui::Color32::from_rgb(179, 38, 30))
                        };
                        ui.add_sized(
                            [46.0, 18.0],
                            egui::Label::new(egui::RichText::new(verdict).color(color).strong()),
                        );
                        let why = r.get(10).map(|v| v.trim_start_matches("否：").to_string()).unwrap_or_default();
                        let note = r.get(11).cloned().unwrap_or_default();
                        let text = if ok {
                            note
                        } else if note.is_empty() {
                            why
                        } else {
                            format!("{why}；{note}")
                        };
                        ui.add(egui::Label::new(egui::RichText::new(&text).color(
                            if ok { ui.visuals().weak_text_color() } else { color },
                        )).truncate())
                        .on_hover_text(&text);
                        ui.end_row();
                    }
                });
        });
    }
}
