//! 两个校验项目共用的底层。
//!
//! - [`analysis`]: 解码 + 各项声学指标的计算。这一层只"测", 不"判"——
//!   哪个指标合不合格是各项目自己的事, 别往这里塞判定。
//! - [`util`]: 跟音频无关的界面外壳零件(字体、输出路径、写 CSV、打开文件)。
//!
//! 判定逻辑各自留在各自的 bin 里:
//!   - 音乐实录(项目A): 本 crate 的 `src/main.rs` + `analysis::check_song`
//!   - 原始素材(项目B): `crates/audio_qc_raw`

pub mod analysis;
pub mod util;
