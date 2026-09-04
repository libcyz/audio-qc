#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]
//! 音频实录质量校验 - 拖拽版 (Rust)
//!
//! 把音频文件或文件夹拖进窗口即可。只校验音频本身, 不看 meta json。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use audio_qc::analysis::{self, check_song, group_audio, SongGroup, AUDIO_EXT, COLUMNS, NOTES_COL};
use audio_qc::util::{install_cjk_font, open_in_system, out_path, write_csv};
use eframe::egui;

const MAX_SCAN_FILES: usize = 5000;   // 防呆: 拖进来一整个盘时别无限扫下去

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // 放得下 12 列(总宽约 1640)又不至于超出常见屏幕
            .with_inner_size([1280.0, 700.0])
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
    cfg: analysis::Settings,
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
        let cfg = self.cfg;
        std::thread::spawn(move || worker(jobs, root, out, tx, cancel, cfg));
    }
}

fn worker(
    jobs: Vec<SongGroup>,
    root: Option<PathBuf>,
    out: PathBuf,
    tx: Sender<Msg>,
    cancel: Arc<AtomicBool>,
    cfg: analysis::Settings,
) {
    let total = jobs.len();
    let mut all: Vec<Vec<String>> = Vec::new();
    let mut bad_songs = 0usize;

    for (i, g) in jobs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let _ = tx.send(Msg::Status(format!("({}/{}) {}", i + 1, total, g.title)));
        let rows = check_song(g, root.as_deref(), &cfg);
        if rows.iter().any(|r| r[analysis::VERDICT_COL] != "是") {
            bad_songs += 1;
        }
        all.extend(rows.clone());
        let _ = tx.send(Msg::Rows(rows, i + 1));
    }

    let (out, err) = match write_csv(&out, &COLUMNS, &all) {
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
                    // 高度固定, 内容行数变化时面板不跟着长高 —— 否则上面板一变高,
                    // 下面的表格视口就变, 又会牵动布局
                    ui.set_min_height(52.0);
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new(drop_text).size(17.0));
                        if !self.hint.is_empty() && !self.busy {
                            ui.label(egui::RichText::new("可以继续拖下一批").weak());
                        }
                    });
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

            // 平均幅值的两个可调项。校验途中不让改, 免得同一批结果用了两套参数。
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.busy, |ui| {
                    ui.label("平均幅值 活动段判据");
                    let cur = match self.cfg.avg_mode {
                        analysis::AvgMode::Energy => "能量阈值",
                        analysis::AvgMode::Vad(0) => "VAD 0 (最宽松)",
                        analysis::AvgMode::Vad(1) => "VAD 1",
                        analysis::AvgMode::Vad(2) => "VAD 2",
                        _ => "VAD 3 (最严)",
                    };
                    egui::ComboBox::from_id_salt("avg_mode")
                        .selected_text(cur)
                        .show_ui(ui, |ui| {
                            use analysis::AvgMode::*;
                            ui.selectable_value(&mut self.cfg.avg_mode, Vad(0), "VAD 0 (最宽松)");
                            ui.selectable_value(&mut self.cfg.avg_mode, Vad(1), "VAD 1");
                            ui.selectable_value(&mut self.cfg.avg_mode, Vad(2), "VAD 2");
                            ui.selectable_value(&mut self.cfg.avg_mode, Vad(3), "VAD 3 (最严)");
                            ui.selectable_value(&mut self.cfg.avg_mode, Energy, "能量阈值");
                        })
                        .response
                        .on_hover_text(
                            "WebRTC VAD 按人声特征挑活动段, 数字越大判得越严。\n\
                             纯伴奏轨 VAD 认不出人声时会自动退回能量阈值。",
                        );

                    ui.add_space(12.0);
                    ui.label("合格范围");
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.avg_db_lo)
                            .speed(0.5)
                            .range(-60.0..=0.0)
                            .suffix(" dBFS"),
                    );
                    ui.label("～");
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.avg_db_hi)
                            .speed(0.5)
                            .range(-60.0..=0.0)
                            .suffix(" dBFS"),
                    );
                    if ui.button("恢复默认").clicked() {
                        self.cfg = analysis::Settings::default();
                    }
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

    /// 每列宽度, 与 analysis::COLUMNS 一一对应 —— 界面显示的字段必须和 csv 完全一致,
    /// 不另起一套简称。全部定宽: truncate() 会去索取"可用宽度", 而在可横向滚动的
    /// ScrollArea 里可用宽度又取决于滚动条在不在, 两者每帧互相追逐, 窗口就会闪。
    const WIDTHS: [f32; 15] = [
        250.0, // 文件名
        48.0,  // 格式
        62.0,  // 采样率
        78.0,  // 时长 (mm:ss.mmm)
        72.0,  // 码率kbps
        72.0,  // 底噪dBFS
        84.0,  // 截至频率Khz
        86.0,  // 人声活动比例
        86.0,  // 伴奏活动比例
        98.0,  // 人声伴奏分贝差
        72.0,  // 峰值电平
        72.0,  // 平均幅值
        84.0,  // 是否削波
        300.0, // 是否满足要求
        320.0, // 备注（如有）
    ];

    // 与 analysis::VERDICT_COL 保持一致 —— 那边用 COLUMNS.len()-2 算, 这里跟着算,
    // 免得又留一个写死的数字。
    const VERDICT: usize = super::COLUMNS.len() - 2;
    const GREEN: egui::Color32 = egui::Color32::from_rgb(26, 127, 55);
    const RED: egui::Color32 = egui::Color32::from_rgb(179, 38, 30);

    pub fn table(ui: &mut egui::Ui, rows: &[Vec<String>]) {
        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("result")
                .striped(true)
                .spacing([10.0, 6.0])
                .show(ui, |ui| {
                    for (h, w) in super::COLUMNS.iter().zip(WIDTHS) {
                        ui.add_sized(
                            [w, 18.0],
                            egui::Label::new(egui::RichText::new(*h).strong()).truncate(),
                        );
                    }
                    ui.end_row();

                    for r in rows {
                        let ok = r.get(VERDICT).map(|v| v == "是").unwrap_or(false);
                        for (i, w) in WIDTHS.iter().enumerate() {
                            let cell = r.get(i).cloned().unwrap_or_default();
                            let mut text = egui::RichText::new(&cell);
                            if i == VERDICT {
                                text = text.color(if ok { GREEN } else { RED }).strong();
                            } else if i == super::NOTES_COL {
                                text = text.color(ui.visuals().weak_text_color());
                            }
                            ui.add_sized([*w, 18.0], egui::Label::new(text).truncate())
                                .on_hover_text(&cell);   // 列窄看不全时鼠标悬停看全文
                        }
                        ui.end_row();
                    }
                });
        });
    }
}
