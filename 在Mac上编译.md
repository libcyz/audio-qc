# 在 macOS 上编译

Windows 上打的 `.exe` 在 Mac 上跑不了（PE 格式，macOS 只认 Mach-O）。
但这份 Rust 源码是跨平台的，在 Mac 上重新编译一次就行。

## 步骤

```bash
# 1. 装 Rust（已装可跳过）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. 把整个 audio_qc 目录拷到 Mac 上，进去编译
cd audio_qc
cargo build --release

# 3. 产物在这里，双击或命令行都行
./target/release/audio_qc
```

首次编译要下载依赖并编译 egui，大约 3~5 分钟；之后增量编译几秒。

## 验证有没有编译对

```bash
cargo test --release     # 6 项算法自检，应全部 ok
./target/release/audio_qc /path/to/音频目录    # 也可直接传路径
```

## 与 Windows 版的差异

代码里只有四处平台相关，已经做成条件编译，不需要你改：

| 位置 | Windows | macOS |
|---|---|---|
| 中文字体 | 微软雅黑/黑体 | 苹方 PingFang / 黑体-简 |
| 打开结果 CSV | `cmd /C start` | `open` |
| 桌面目录（只读盘时的退路） | `%USERPROFILE%\Desktop` | `$HOME/Desktop` |
| 隐藏控制台窗口 | `windows_subsystem` | 不需要 |

判定逻辑、阈值、CSV 表头完全一样，同一个音频在两个平台上结果应当一致
（都是纯 Rust 的 f64 运算，没有平台相关的浮点差异）。

## 注意

- 首次打开可能被 Gatekeeper 拦（"无法验证开发者"）。右键点图标选"打开"，
  或者 `xattr -d com.apple.quarantine ./target/release/audio_qc`。
  自己编译的二进制没有签名，这是正常的。
- 想要 .app 图标包或分发给别人，需要 Apple 开发者证书做签名和公证，
  自用的话直接跑上面的二进制就够了。

## 交叉编译说明

不能在 Windows 上直接编译出 Mac 版——那需要 Apple 的 SDK 和链接器，
苹果的许可也不允许。必须在 Mac 上编译。
