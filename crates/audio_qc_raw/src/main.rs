#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]
//! 原始录音素材校验 (项目B) - 拖拽版
//!
//! 跟"音频实录质量校验"(项目A) 是两个独立的程序: 校验项、阈值、CSV 列都不一样,
//! 只共用 audio_qc 里的解码和 DSP。一个文件一行, 没有按歌分组这回事。

mod check;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use audio_qc::analysis::AUDIO_EXT;
use audio_qc::util::{install_cjk_font, open_in_system, out_path, write_csv};
use check::{check_file, COLUMNS, NOTES_COL, VERDICT_COL};
use eframe::egui;

const MAX_SCAN_FILES: usize = 5000;   // 防呆: 拖进来一整个盘时别无限扫下去
const TITLE: &str = "原始录音素材校验";

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 700.0])
            .with_min_inner_size([760.0, 480.0])
            .with_drag_and_drop(true)
            .with_title(TITLE),
        ..Default::default()
    };
    eframe::run_native(
        TITLE,
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
    Row(Vec<String>, usize),
    Done { out: Option<PathBuf>, files: usize, bad: usize, err: Option<String> },
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
    cfg: check::Settings,
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
        let missing = paths.iter().filter(|p| !p.exists()).count();
        if missing > 0 {
            self.toast = format!("有 {missing} 个路径不存在，已跳过");
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
    jobs: Vec<PathBuf>,
    root: Option<PathBuf>,
    out: PathBuf,
    tx: Sender<Msg>,
    cancel: Arc<AtomicBool>,
    cfg: check::Settings,
) {
    let total = jobs.len();
    let mut all: Vec<Vec<String>> = Vec::new();
    let mut bad = 0usize;

    for (i, p) in jobs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let shown = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        let _ = tx.send(Msg::Status(format!("({}/{}) {}", i + 1, total, shown)));
        let row = check_file(p, root.as_deref(), &cfg);
        if row[VERDICT_COL] != "是" {
            bad += 1;
        }
        all.push(row.clone());
        let _ = tx.send(Msg::Row(row, i + 1));
    }

    let (out, err) = match write_csv(&out, &COLUMNS, &all) {
        Ok(p) => (Some(p), None),
        Err(e) => (None, Some(e)),
    };
    let _ = tx.send(Msg::Done { out, files: all.len(), bad, err });
}

/// 递归收集音频。项目B 一个文件一行, 不按歌名分组。
///
/// 不只收 .wav —— 拖进来一个 mp3 也要走完流程, 在"文件类型"列如实写 MP3 并判不合格,
/// 而不是当成"没找到文件"静静跳过。那样交付方会以为这批全过了。
fn collect(paths: &[PathBuf]) -> (Vec<PathBuf>, bool) {
    let mut jobs = Vec::new();
    let mut truncated = false;

    let is_audio = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .map(|e| AUDIO_EXT.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    };

    for p in paths {
        if p.is_dir() {
            for entry in walkdir::WalkDir::new(p).follow_links(false).into_iter().flatten() {
                if jobs.len() >= MAX_SCAN_FILES {
                    truncated = true;
                    break;
                }
                let f = entry.path();
                if f.is_file() && is_audio(f) {
                    jobs.push(f.to_path_buf());
                }
            }
        } else if is_audio(p) {
            jobs.push(p.clone());
        }
    }
    jobs.sort();
    (jobs, truncated)
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let ctx = &ctx;
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
                    Msg::Row(row, done) => {
                        self.rows.push(row);
                        self.progress.0 = done;
                    }
                    Msg::Done { out, files, bad, err } => {
                        self.busy = false;
                        self.csv = out.clone();
                        self.hint = format!("{files} 个文件，不合格 {bad} 个");
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
                    // 高度固定, 内容行数变化时面板不跟着长高
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
                if ui.add_enabled(self.csv.is_some(), egui::Button::new("打开结果CSV")).clicked() {
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

            // 可调项。校验途中不让改, 免得同一批结果用了两套参数。
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.busy, |ui| {
                    ui.label("信噪比下限");
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.min_snr_db)
                            .speed(0.5)
                            .range(0.0..=120.0)
                            .suffix(" dB"),
                    )
                    .on_hover_text("信噪比 = 活动段 RMS − 底噪。底噪按静音帧的能量平均算, 没有静音段的文件测不出。");

                    ui.add_space(12.0);
                    ui.label("静音比例上限");
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.max_silence)
                            .speed(0.01)
                            .range(0.0..=1.0)
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                    )
                    .on_hover_text("静音判据沿用能量门限: 低于本轨 P95 电平 −35dB 且低于 −55dBFS 的帧。");

                    ui.add_space(12.0);
                    ui.label("时长下限");
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.min_dur_s)
                            .speed(1.0)
                            .range(0.0..=3600.0)
                            .suffix(" 秒"),
                    );
                    if ui.button("恢复默认").clicked() {
                        self.cfg = check::Settings::default();
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

/// 结果表格。跟项目A 一样手写定宽表: truncate() 会去索取"可用宽度", 而在可横向
/// 滚动的 ScrollArea 里可用宽度又取决于滚动条在不在, 两者每帧互相追逐, 窗口会闪。
mod result_table {
    use eframe::egui;

    /// 每列宽度, 与 check::COLUMNS 一一对应 —— 界面显示的字段必须和 csv 完全一致。
    const WIDTHS: [f32; super::COLUMNS.len()] = [
        260.0, // 文件名
        150.0, // 文件类型
        86.0,  // 是否立体声
        56.0,  // 声道数
        62.0,  // 采样率
        62.0,  // 采样位深
        56.0,  // 时长
        66.0,  // 静音比例
        68.0,  // 信噪比dB
        64.0,  // 响度dB
        120.0, // 频响
        320.0, // 是否满足要求
        380.0, // 备注（如有）
    ];

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
                        let ok = r.get(super::VERDICT_COL).map(|v| v == "是").unwrap_or(false);
                        for (i, w) in WIDTHS.iter().enumerate() {
                            let cell = r.get(i).cloned().unwrap_or_default();
                            let mut text = egui::RichText::new(&cell);
                            if i == super::VERDICT_COL {
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
