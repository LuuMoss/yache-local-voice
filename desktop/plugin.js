// Yache Voice — 雅澈语音 status-bar companion (M1)
// Pattern mirrors the official radio plugin (pure ESM hot-load door, no JSX):
// status-bar chip -> Popover quick panel (doctor lamps + VRAM + Done; a pure
// health window — audition lives in the cloning modal now), plus a settings
// console under Settings ▸ Plugins: doctor lamps inline, four section entry
// cards with live summaries, each opening a ~640px modal with an explicit
// Back (explicit-exit rule; save-with-self-check lands per section as its
// parameters get wired next).
import { Button, ConfirmDialog, Input, Popover, PopoverContent, PopoverTrigger, Select, SelectContent,
         SelectItem, SelectTrigger, SelectValue, RowButton,
         STATUSBAR_AREAS, PALETTE_AREA, Tip, cn, icons, usePluginI18n,
         pluginSettingsHref, queryClient, Codicon } from '@hermes/plugin-sdk';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Fragment, jsx, jsxs } from 'react/jsx-runtime';
import { host } from '@hermes/plugin-sdk';

const ID = 'yache-local-voice';
const API = '/api/plugins/' + ID;

// Privacy masking for displayed paths: swap the Windows username segment for
// a reader-friendly placeholder (owner ruling 2026-11-19), so neither a
// maintainer's screenshots nor an end user's bug reports leak account names.
// Only the account name changes; drive letters and folder layout stay real,
// so the line remains a usable diagnostic.  The /g flag is load-bearing: the
// engine env line packs several such paths into one string (python= and
// qwen.dll=); a non-global replace leaks every path after the first.
function maskPath(p, who) {
  return String(p || '').replace(/([A-Za-z]:[\\/]Users[\\/])[^\\/]+/g, '$1' + who);
}

// —— file log pipe (mirrors video-xray's proven /debug design, 2026-11-18) —
// Whatever the panel learns but cannot show is POSTed to the backend /debug
// route, which appends it to the plugin's own yache-plugin.log — so a future
// "failed to render" leaves more than a white error page behind. The ctx
// captured at register() feeds the pipe; everything is fire-and-forget and
// logging must NEVER be the thing that breaks the panel.
let _api = null;
function dbg(where, detail) {
  let msg = String(where);
  try {
    if (detail !== undefined && detail !== null) {
      const d = (detail && detail.stack) || (detail && detail.message) ||
        (typeof detail === 'object' ? JSON.stringify(detail) : String(detail));
      msg += ' | ' + d;
    }
  } catch (e) { msg += ' | <unprintable>'; }
  try { if (typeof console !== 'undefined') console.debug('[yache-dbg] ' + msg); } catch (e) {}
  const c = _api;
  if (!c) return;
  try { c.rest('/debug', { method: 'POST', body: { msg }, timeoutMs: 2500 }).catch(() => {}); }
  catch (e) { /* diagnostics never raise */ }
}
function hookGlobalErrorNet() {
  if (typeof window === 'undefined' || window.__yacheVoiceDbgNet) return;
  window.__yacheVoiceDbgNet = true;
  window.addEventListener('error',
    (ev) => dbg('window.error', (ev && (ev.error || ev.message)) || 'unknown'));
  window.addEventListener('unhandledrejection',
    (ev) => dbg('unhandledrejection', (ev && ev.reason) || 'unknown'));
}

const LOCALES = {
  // nested bundle, NOT flat dotted keys: the host i18next splits '.'
  // into a lookup path, so t('yache.brand') needs bundle.yache.brand
  // (the shape the official radio plugin and video-xray both ship).
  zh: { yache: {
      brand: '雅澈语音',
      tip: '语音引擎状态与体检',
      refresh: '重新体检',
      settings: '高级设置',
      done: '完成 ✓',
      done_title: '关闭面板',
      back: '← 返回',
      doctor: '六灯体检',
      lamp: {
        plugin: '插件装载',
        engine: '引擎环境',
        models: '模型文件',
        vram: '显存余量',
        voice: '音色库',
        worker: '合成工人',
      },
      vram: '空闲显存',
      vram_none: '无 NVIDIA 显卡（CPU 模式）',
      rec: '推荐档位',
      selected: '当前档位',
      test: '试听（默认音色，首次合成约 5 秒）',
      test_ph: '留空则用示例句',
      test_go: '▶ 试听',
      test_busy: '合成中…',
      test_fail: '试听失败',
      page_title: '雅澈语音 · 语音链路控制台',
      page_desc: '本地 TTS / STT / 唤醒词全链路。主页看体检灯与四个分区概况，每区点「设置」进弹窗细调：克隆语音（选模型＋建音色，TTS 由此解锁）、TTS 语音合成（档位随音色自动读取）、唤醒与对话（会话方式、停止词、念读闸门）、STT 语音转录（即将上线）。',
      model_dir: '模型目录',
      voices: '音色库',
      stop_title: '对话停止词',
      stop_desc: '语音对话中说出一个整句、与停止词逐字相同，本次对话即结束（官方整句匹配语义，句子里顺带提及不会误停）。需唤醒词开关处于开启状态；保存后立即可说，无需重启。',
      stop_ph: '如：结束会话（多个停止词用逗号分隔）',
      stop_default: '当前使用官方默认词',
      stop_save: '保存',
      stop_saving: '保存中…',
      stop_saved: '已保存，立即生效 ✓',
      stop_fail: '保存失败',
      wake_title: '唤醒后会话方式',
      wake_desc: '选择每次唤醒后的行为。「开新会话」＝每次喊醒都开一个新聊天窗口（官方默认，Hey Siri 式）；「继续当前会话」＝唤醒 → 对话 → 说停止词暂停 → 再唤醒接着当前窗口聊，历史记录全部保留（写官方键 wake_word.start_new_session）。官方在唤醒开始监听时读取该值：保存后把唤醒关掉再打开一次（或重启网关）即切到新设置。需唤醒词处于开启状态。',
      wake_opt_new: '开新会话（官方默认）',
      wake_opt_keep: '继续当前会话',
      wake_save: '保存',
      wake_saving: '保存中…',
      wake_saved: '已保存：下次唤醒布防起生效',
      wake_fail: '保存失败',
      wake_default: '当前为官方默认（开新会话）',
      wp_title: '唤醒词（中英双语）',
      wp_desc: '喊哪个词就用哪个词唤醒，中文、英文都支持。保存时按词的文字自动挑选本机匹配的模型（中文→拼音声学模型，英文→拼写模型），走官方构造器现场真机加载验证：验不过不保存，引擎报错原样显示。建议 2~6 个汉字或 1~3 个英文单词，太长识别率会降。正在监听时，新词要把唤醒开关关→开一次才生效；未监听则下次开启唤醒自动生效。',
      wp_label: '输入唤醒词，如 小美 / hello hermes',
      wp_save: '保存',
      wp_verifying: '真机加载验证中…（约几秒）',
      wp_saved_auto: '已保存，下次开启唤醒即用新词 ✓',
      wp_saved_rearm: '已保存 —— 正在监听中，把唤醒开关关→开一次即用新词 ✓',
      wp_model_switched: '（已自动切换到匹配该语言的本机模型）',
      wp_name_sync: '（提醒：唤醒词不会自动改助手的自称，想名字一致请在你的档案/记忆设置里同步）',
      wp_rejected: '引擎拒绝，未保存：',
      wp_now: (kind, st) => `当前底层：${kind} · ${st}`,
      wp_kind_zh: '拼音模型（中文）',
      wp_kind_en: '拼写模型（英文）',
      wp_kind_unknown: '未知',
      wp_armed: '监听中',
      wp_idle: '未监听',
      wp_mixed_note: '（当前词中英混排，能否命中以真机验证为准）',
      gate: '念读闸门（实验）',
      gate_tip: '念读回复期间暂停唤醒检测；默认关闭 = 官方行为（对话由停止词结束）。想先体验可在 yache-voice.json 加 "wake_gate": true。',
      status_serving: '正在服务',
      status_ready: '已就绪·未生效',
      status_notready: '未就绪',
      badge_tts: '合成', badge_stt: '转录', badge_wake: '唤醒',
      badge_dormant: '未生效', btn_dormant: '未生效',
      dormant_tip: '该语音通道正被别的 provider 占用 — 本地引擎零驻留休息，随时一键切回',
      cloud_tts_detail: (p) => `合成未生效 — ${p} 在用（雅澈引擎已就绪待命，零驻留，官方入口照常归它管）`,
      cloud_stt_detail: (p) => `转录未生效 — ${p} 在用（本地引擎已就绪待命）`,
      cloud_clone_detail: (p) => `克隆音色未生效 — 合成走的是 ${p}`,
      switch_back: '切回本地',
      switching: '验证切换中（可能要下载模型，请稍候）…',
      switch_ok: '已切回本地 provider，下次出声即由本地引擎',
      switch_ok_stt: '已切回本地 provider，下次开口即由本地转录',
      switch_failed: '切回本地失败',
      refreshing: '检查中…',
      refresh_done: '体检完成，六盏灯已刷新',
      refresh_failed: '体检请求失败，稍后再试',
      wake_off_card: '官方唤醒已关闭 — 去官方设置打开即可（不挡路）',
      sec_open: '设置',
      sec_tts: 'TTS 语音合成',
      sec_stt: 'STT 语音转录',
      sec_wake: '唤醒与对话',
      sec_clone: '克隆语音',
      sec_locked: '未解锁',
      sec_locked_tip: '先完成音色克隆（选模型＋建音色＋试听保存），TTS 语音合成才会解锁。',
      soon: '即将上线',
      stt_desc: '转录引擎 = faster-whisper（官方单一路径，不走镜像）。保存前引擎会真正加载所选模型验证——未缓存的尺寸当场从官方源下载——引擎点头才落账；失败原话回显、旧设置不动。',
      stt_loading: '正在体检转录引擎…',
      stt_engine_down: '转录引擎不可用',
      stt_engine_tip: '先把下面的 ASR 环境装好（插件自有，一键安装），再点重新体检。',
      stt_env_title: 'ASR 环境',
      stt_env_check: '正在检查环境…',
      stt_env_ready: '已就绪',
      stt_env_missing: '未安装',
      stt_env_deps: '缺引擎依赖',
      stt_env_broken: '已失效',
      wiz_top_tip: '本地没有模型？→ 联网下载（下方向导点几下就好）；本地已有模型？→ 直接打开模型目录装载。',
      wiz_title: '联网下载',
      wiz_model: '模型',
      wiz_quant_l: '量化',
      wiz_size: (a, b) => `合计下载约 ${a}，运行需要内存/显存约 ${b}；没有独立显卡也能用 CPU 出声`,
      wiz_src: (a) => `下载源：${a}（自动探测最快）`,
      wiz_src_probing: '正在探测最快下载源…（首次约 5 秒）',
      wiz_src_fail: 'huggingface.co 与镜像都连不上——请开启代理/VPN 后点“重测”。',
      wiz_probe: '重测',
      wiz_dir: '保存到',
      wiz_dir_ph: '选一个文件夹即可（自动创建 models 子目录，已有文件不受影响）',
      wiz_browse: '浏览…',
      wiz_start: '开始下载',
      wiz_cancel: '取消',
      wiz_cancelling: '取消中…',
      wiz_resume: '继续下载（断点续传）',
      wiz_prog: (p, m, e) => `${p}% · ${m} MB/s · 剩余 ${e}`,
      wiz_eta_calc: '计算中',
      wiz_done: '下载完成，已自动装载进本地语音。',
      wiz_gpu_done: '加速包已装好——检测到 CUDA 引擎。',
      wiz_extracting: '正在解压部署加速引擎…（几十秒，请稍候）',
      wiz_uninstall: '卸载加速包',
wiz_uninstall_confirm: '确认卸载？（8 个 CUDA DLL，几秒完成）',
wiz_uninstalled: 'CUDA 引擎已卸载——需要时点下方按钮随时装回。',
      wiz_uninstall_busy: '下载向导正在运行，等它结束再卸载。',
      wiz_uninstall_in_use: '语音合成正在使用 CUDA 引擎，请待合成结束后再试。',
      wiz_uninstall_not_pack: '该目录不像本插件装的加速包，已保护性跳过（未删除任何文件）。',
      wiz_uninstall_io: '卸载失败（文件被占用或权限不足），请稍后重试。',
      wiz_gpu_title: 'GPU 加速包（可选 · 提速用）',
      wiz_gpu_on: '已安装 · CUDA 提速生效中',
      wiz_gpu_off: '未安装 · CPU 也能出声；装上后快很多。',
      wiz_gpu_nvidia: '仅 NVIDIA 显卡可提速；AMD / 核显用户保持默认 CPU 模式即可，念读照常工作。',
      wiz_gpu_btn: (gb) => `下载 GPU 加速包（约 ${gb}）`,
      wiz_gpu_note: '约 750MB，来自 GitHub（自动挑最快源，直连失败走国内加速镜像），下载后自动解压并校验，下次念读自动提速。',
      wiz_gpu_locked: '先把模型下好或装载好，这个按钮才会亮',
      path_who: '用户名',
      stt_env_ready_desc: (py) => `已实测能加载的引擎环境：${py}`,
      stt_env_missing_desc: '还没装引擎环境。下面这个是插件自有的独立 Python 环境，放在 Hermes 数据目录里，官方升级 Hermes 不会碰它；点一下就装好。',
      stt_env_deps_desc: '环境还在，可里面的转录引擎不见了——多半是某次环境重建把它清掉了（这类引擎不在 Hermes 依赖锁里）。点一下补回来，模型不用重下。',
      stt_env_broken_desc: '环境文件夹还在，但它的 Python 已经起不来了（宿主的 Python 被换过，或上次装到一半被打断）。点一下重建即可。',
      stt_env_install: '安装 ASR 环境',
      stt_env_rebuild: '重装环境',
      stt_env_wipe: '清空旧环境',
      stt_env_venv: '创建虚拟环境',
      stt_env_pip: '安装转录引擎',
      stt_env_verify: '验证引擎加载',
      stt_env_prog: (s, pct, el) => `${s} · ${pct}% · 已用 ${el}s`,
      stt_env_done: 'ASR 环境已就绪',
      stt_env_fail: '环境安装失败：',
      stt_env_unfail: '环境卸载失败：',
      open_log: '打开日志',
      open_log_fail: '打开日志失败：',
      stt_env_uninstall: '卸载环境',
      stt_env_uninstall_confirm: '再点一次确认卸载',
      stt_env_uninstalled: 'ASR 环境已卸下——模型缓存仍保留，想清盘子可在模型区单独删除。',
      stt_env_disk: (gb) => `提醒：目标磁盘只剩 ${gb} GB，环境加模型至少要 1.5 GB，先清一清再装。`,
      stt_env_other: (py) => `注：当前生效的其实是外部 Python：${py}`,
      stt_retry: '重新体检',
      stt_model: '模型',
      stt_cached: '已在缓存',
      stt_dl: '本地未缓存 · 可在线下载',
      stt_dir_entry: '本地模型目录…',
      stt_dir_label: '本地模型目录（需 CTranslate2 格式，含 model.bin / tokenizer.json）',
      stt_dir_browse: '选择…',
      stt_device: '计算设备',
      stt_dev_auto: '自动（有 GPU 优先 GPU）',
      stt_lang: '语言',
      stt_lang_auto: '自动检测',
      stt_save: '验证并保存',
      stt_save_locked: '选了未下载的模型：先按上面「下载此模型」，下载到位后这个按钮才会解锁',
      stt_dl_btn: (gb) => `下载此模型（约 ${gb} GB，可能要几分钟到几十分钟）`,
      stt_dl_busy: '下载中，请勿关闭窗口…',
      stt_dl_hint: (gb) => `这个模型本地还没有（约 ${gb} GB）：先点下面的「下载此模型」完整下到本地——下载失败=网络/代理问题，保存失败=引擎问题，两步各自说清。`,
      stt_dl_ok: (m, s) => `下载完成并已真加载验证：${m} 已在本地缓存（${s} 秒加载），「验证并保存」已解锁。`,
      stt_dl_cached: (m) => `${m} 本来就在本地缓存里，直接点「验证并保存」即可。`,
      stt_dl_fail: '下载失败（网络/代理问题？）：',
      stt_dl_done: (m) => `下载完成：${m} 已在本地缓存 — 「验证并保存」已解锁，可以点了。`,
      stt_dl_prog: (p, r, e) => `${p}% · ${r} MB/s · 预计还需 ${e}`,
      stt_dl_eta_calc: '剩余时间计算中…',
      stt_del_btn: (gb) => `删除此模型（释放约 ${gb} GB）`,
      stt_del_active: (m) => `${m} 正在生效 — 先切到别的模型保存后，再来删它`,
      stt_del_title: '删除语音转录模型',
      stt_del_desc: (m, gb) => `将把 ${m} 的本地缓存（约 ${gb} GB）从磁盘彻底删除。想再用时点「下载此模型」重新取回即可，不保留任何记录。`,
      stt_del_confirm: '删除',
      stt_del_busy: '删除中…',
      stt_del_done: '已删除',
      stt_del_ok: (m, gb) => `已删除 ${m}，释放约 ${gb} GB。`,
      stt_verifying: '引擎从本地缓存真加载验证中…（几秒）',
      stt_saved: (m, d, s) => `已保存并验证通过：${m} · 设备 ${d} · ${s} 秒加载完成，即刻生效。`,
      stt_rejected: '保存失败（旧设置未改动）：',
      stt_need_model: '请先选择模型尺寸，或填写本地模型目录',
      stt_now: (m, d, l) => `当前生效：${m || '—'} · 设备 ${d} · 语言 ${l}`,
      soon_clone: '音色管理与「新建克隆」（参考音频＋文本，保存即真机合成验证）即将上线。',
      clone_new: '新建克隆',
      voice_unset: '音色未设置',
      models_missing: (list) => '缺少模型文件：' + list + '（下载地址见插件目录 README）',
      tokenizer_missing: 'Tokenizer 缺失（models/ 下缺 tokenizer.json）',
      mdl_title: '模型目录',
      mdl_hint: '告诉雅澈模型放在哪里：选包含 qwen-talker-1.7b-*.gguf 的文件夹（选外层目录或 models 子目录都可以，会自动识别）。「读取」会真实扫描该目录，扫描通过才会生效；引擎每句话都重读目录，切换即时生效、无需重启。',
      mdl_ph: '模型目录路径（也可以直接粘贴）',
      mdl_fresh: '模型还没就位——用上方下载中心挑个您想要的目录下载（不必装在 C 盘），下完会自动装载到这里；本地已有模型的话点「选目录」指过去再点「读取」装载',
      mdl_pick: '选目录',
      mdl_read: '读取',
      mdl_reset: '恢复默认',
      mdl_ok: (n) => '目录已应用：找到 ' + n + ' 个模型变体',
      mdl_reset_ok: '已恢复默认目录',
      mdl_dirty: '目录路径已修改但还没确认——点「读取」扫描后才会列出该目录里的模型',
      vt_title: '模型变体',
      vt_ph: '请先选择一个变体',
      vt_base: 'Base · 用参考音频克隆真人的声音',
      vt_cv: 'CustomVoice · 内置音色挑一个',
      vt_vd: 'VoiceDesign · 用文字描述生成声音',
      vt_hint: '选好模型目录后点「读取」；扫描识别出变体后，这里才会出现参数菜单。',
      vt_pick_hint: '先在上方选择一个模型变体',
      scan_err: (e) => '当前模型目录不可用：' + e,
      grp_spk: (n) => '内置音色 · ' + n,
      grp_cln: (n) => '克隆音色 · ' + n,
      grp_design: '设计音色（文字造声）',
      pick_voice: '选择音色',
      design_desc: '声音描述',
      design_ph: '例：20 岁出头的温柔女声，带笑意，语速偏慢',
      voice_save: '保存音色',
      voice_saving: '保存中…（真实合成验证）',
      pick_applied_clone: (n) => `\u2705 已选择 ${n} 克隆音色，将应用于TTS语音合成音色播放！`,
      pick_applied_spk: (n) => `\u2705 已选择 ${n} 内置音色，将应用于TTS语音合成音色播放！`,
      clone_next_step: '下一步：回总览打开「TTS 语音合成」分区的设置，挑好合成档位保存，助手就会用这个声音念给您听。',
      voice_saved: '音色已保存并设为当前 ✓ TTS 合成已解锁',
      dz_export: '导出试听音频',
      dz_export_tip: '描述造声每次都是随机的新嗓子，导出才能留住刚才这段声音',
      dz_adopt: '定为克隆底声',
      dz_adopt_tip: '给它起个名字，刚才试听的嗓子就固定成克隆音色，以后每次念读都用它（命名只允许中英文和数字下划线）',
      dz_name_ph: '先给这个声音起名',
      dz_need_name: '请先给这个声音起个名字。',
      dz_need_clip: '先点「试听」，才有可导出的声音。',
      dz_ready: '这段试听已就绪：可导出留档，或起名后定为克隆底声。',
      dz_saved: '已导出：',
      dz_adopted: '克隆音色已注册并设为当前：',
      dz_drop: '移走这个克隆底声',
      dz_drop_confirm: '再点一次确定移走',
      dz_drop_tip: '从注册表移走它；磁盘上的录音文件保留不动，想彻底删自己去删',
      dz_dropped: '已移走：',
      dz_dropped_wav: '（录音文件仍留在磁盘）',
      voice_fail: '失败：',
      aud_ok: (s, e) => '试听完成：' + s + ' 秒音频，合成耗时 ' + e + ' 秒',
      clone_new_title: '新建克隆音色',
      clone_name_ph: '音色名（字母数字下划线或中文）',
      clone_ref_ph: '参考音频路径（wav/mp3，放音色库目录可直接写文件名）',
      clone_ref_pick: '选文件…',
      clone_text_ph: '参考音频说了什么（可空，自动识别）',
      clone_desc_ph: '音色说明（可空）',
      clone_go: '创建并验证',
      ref_asr_busy: '正在识别参考音频里的语音，请稍候…',
      ref_asr_ok: '已自动识别出参考文本，请核对，可直接手动修改。',
      ref_asr_fail: '没能识别出内容，请在下方手动输入参考音频里说的原话。',
      clone_none: '还没有克隆音色——点下方“新建克隆音色”，选择本机参考音频文件，系统会自动识别台词，核对无误后保存即用',
      q_auto: '智能（按空闲显存自动选）',
      q_q8_0: '高画质 · Q8_0',
      q_q4_k_m: '省显存 · Q4_K_M',
      q_bf16: '原精度 · BF16',
      q_missing: '已挪走',
      d_auto: '自动（GPU 优先）',
      d_cuda: 'GPU（CUDA）',
      d_cpu: 'CPU',
      tts_quant: '量化档位',
      tts_device: '计算后端',
      tts_save: '保存合成配置',
      tts_saving: '验证中…（用新档位真实合成一次）',
      tts_dirty_tip: '变更尚未保存：点左边按钮保存并验证。',
      verify_failed: '❌ 验证失败：',
      verify_unknown: '后端未给出具体原因（可看插件日志）。',
      tts_saved: '已保存，下次对话念读即用 ✓',
      cur_voice: '当前音色',
      rec_tier: '推荐（按当前音色+空闲显存）',
      cur_backend: '引擎实际使用',
      tts_free_vram: '空闲显存',
    },
  },
  en: { yache: {
      brand: 'Yache Voice',
      tip: 'Voice engine status & doctor',
      refresh: 'Re-run doctor',
      settings: 'Advanced settings',
      done: 'Done ✓',
      done_title: 'Close panel',
      back: '← Back',
      doctor: 'Doctor checks',
      lamp: {
        plugin: 'Plugin loaded',
        engine: 'Engine env',
        models: 'Model files',
        vram: 'Free VRAM',
        voice: 'Voice library',
        worker: 'Synth worker',
      },
      vram: 'Free VRAM',
      vram_none: 'No NVIDIA GPU (CPU mode)',
      rec: 'Recommended tier',
      selected: 'Selected tier',
      test: 'Audition (default voice; first run ~5s)',
      test_ph: 'Leave empty for sample line',
      test_go: '▶ Audition',
      test_busy: 'Synthesizing…',
      test_fail: 'Audition failed',
      page_title: 'Yache Voice · Pipeline Console',
      page_desc: 'The local TTS / STT / wake-word pipeline. The home page shows the doctor lamps and four section summaries; every section opens a modal: Voice cloning (pick a model and create a voice — this unlocks TTS), TTS synthesis (the tier follows the saved voice automatically), Wake & conversation (session mode, stop phrases, playback gate), and STT transcription (arriving next).',
      model_dir: 'Model dir',
      voices: 'Voice library',
      stop_title: 'Stop phrase',
      stop_desc: 'A spoken sentence that matches the stop phrase exactly ends the voice conversation (stock whole-sentence matching — a casual mention inside a longer sentence never fires it). Requires the wake-word switch to be on; takes effect the moment it is saved, no restart.',
      stop_ph: 'e.g. end conversation (comma-separate several)',
      stop_default: 'Official default in use',
      stop_save: 'Save',
      stop_saving: 'Saving…',
      stop_saved: 'Saved — now active ✓',
      stop_fail: 'Save failed',
      wake_title: 'Wake-up conversation mode',
      wake_desc: 'What each wake should do. "New session per wake" opens a fresh chat window every time (stock default, Hey-Siri style); "Continue current session" keeps one conversation — wake, talk, say the stop phrase to pause, and the next wake resumes this same window with the history intact (writes the stock key wake_word.start_new_session). The host reads this value when the wake listener arms: after saving, toggle wake off and on once (or restart the gateway) to apply. Requires the wake word to be on.',
      wake_opt_new: 'New session per wake (stock default)',
      wake_opt_keep: 'Continue current session',
      wake_save: 'Save',
      wake_saving: 'Saving…',
      wake_saved: 'Saved — applies at the next wake arming',
      wake_fail: 'Save failed',
      wake_default: 'Stock default: new session',
      wp_title: 'Wake word (bilingual)',
      wp_desc: 'Whichever word you say is the word that wakes it — Chinese and English both work. Saving auto-picks the local model matching the phrase script (CJK → pinyin acoustic model, latin → spelled model) and loads the real engine right there as a verification through the stock builder: a rejected save changes nothing and the engine error shows verbatim. 2–6 Han characters or 1–3 English words detect best; long phrases detect worse. While armed, toggle wake off→on once to switch words; while idle the next wake start simply uses it.',
      wp_label: 'Type a wake word, e.g. coco / hello hermes',
      wp_save: 'Save',
      wp_verifying: 'Verifying on the real engine… (a few seconds)',
      wp_saved_auto: 'Saved — the next wake start uses the new word ✓',
      wp_saved_rearm: 'Saved — armed right now; toggle wake off→on once to switch ✓',
      wp_model_switched: ' (switched to the local model matching that script)',
      wp_name_sync: ' (note: the wake word does not rename the assistant — sync it in your profile/memory settings if you want them to match)',
      wp_rejected: 'Rejected by the engine, nothing saved:',
      wp_now: (kind, st) => `Engine now: ${kind} · ${st}`,
      wp_kind_zh: 'pinyin model (Chinese)',
      wp_kind_en: 'spelled model (English)',
      wp_kind_unknown: 'unknown',
      wp_armed: 'armed',
      wp_idle: 'idle',
      wp_mixed_note: ' (mixed-script phrase: on-device verification decides)',
      gate: 'Playback gate (exp.)',
      gate_tip: 'Pauses wake detection while replies are read aloud; off by default = stock behavior (conversations end via the stop phrase). Opt in locally with "wake_gate": true in yache-voice.json.',
      status_serving: 'Serving',
      status_ready: 'Ready · dormant',
      status_notready: 'Not ready',
      badge_tts: 'TTS', badge_stt: 'STT', badge_wake: 'Wake',
      badge_dormant: 'dormant', btn_dormant: 'Dormant',
      dormant_tip: 'Another voice provider is in use; local engines rest with zero residency. Switch back any time.',
      cloud_tts_detail: (p) => `Synthesis dormant — ${p} is in use (Yache engine ready, zero residency; the official entry point keeps control)`,
      cloud_stt_detail: (p) => `Transcription dormant — ${p} is in use (local engine ready)`,
      cloud_clone_detail: (p) => `Cloned voices dormant — synthesis is routed to ${p}`,
      switch_back: 'Switch back',
      switching: 'Verifying switch (may download a model, hold on)…',
      switch_ok: 'Back on the local provider — the next reply speaks locally',
      switch_ok_stt: 'Back on the local provider — the next utterance transcribes locally',
      switch_failed: 'Switch back failed',
      refreshing: 'Checking…',
      refresh_done: 'Check complete — the lights are up to date',
      refresh_failed: 'Check request failed, try again shortly',
      wake_off_card: 'Official wake is switched off — enable it in official settings (we never block it)',
      sec_open: 'Open',
      sec_tts: 'TTS synthesis',
      sec_stt: 'STT transcription',
      sec_wake: 'Wake & conversation',
      sec_clone: 'Voice cloning',
      sec_locked: 'Locked',
      sec_locked_tip: 'Finish voice cloning (pick a model, create a voice, audition + save) to unlock TTS synthesis settings.',
      soon: 'arriving next',
      stt_desc: 'Transcription engine = faster-whisper (one official path, no mirrors). Saving really loads the chosen model first (uncached sizes download from the official source on the spot); nothing persists until the engine nods. Errors come back verbatim; old settings stay untouched.',
      stt_loading: 'Checking the transcription engine…',
      stt_engine_down: 'Transcription engine unavailable',
      stt_engine_tip: 'Install the ASR environment below first (plugin-owned, one click), then re-check.',
      stt_env_title: 'ASR environment',
      stt_env_check: 'checking the environment…',
      stt_env_ready: 'ready',
      stt_env_missing: 'not installed',
      stt_env_deps: 'engine packages missing',
      stt_env_broken: 'broken',
      wiz_top_tip: 'No local model? → download it below (a few clicks). Already have one? → open its folder and load it.',
      wiz_title: 'Download',
      wiz_model: 'Model',
      wiz_quant_l: 'Quantization',
      wiz_size: (a, b) => `~${a} to download, ~${b} of RAM/VRAM to run; a CPU-only machine works too`,
      wiz_src: (a) => `Source: ${a} (fastest, auto-detected)`,
      wiz_src_probing: 'Probing the fastest download source… (~5 s)',
      wiz_src_fail: 'Neither huggingface.co nor the mirror is reachable — enable your proxy/VPN and hit Retry.',
      wiz_probe: 'Retry',
      wiz_dir: 'Save to',
      wiz_dir_ph: 'any folder works (a models\\ subfolder is created; existing files are untouched)',
      wiz_browse: 'Browse…',
      wiz_start: 'Start download',
      wiz_cancel: 'Cancel',
      wiz_cancelling: 'Cancelling…',
      wiz_resume: 'Resume download',
      wiz_prog: (p, m, e) => `${p}% · ${m} MB/s · ETA ${e}`,
      wiz_eta_calc: 'calculating',
      wiz_done: 'Downloaded and auto-loaded into local voice.',
      wiz_gpu_done: 'Accelerator pack installed — CUDA engine detected.',
      wiz_extracting: 'Unpacking and deploying the CUDA engine… (a minute or so)',
      wiz_uninstall: 'Uninstall accelerator pack',
wiz_uninstall_confirm: 'Confirm uninstall? (8 CUDA DLLs, a few seconds)',
wiz_uninstalled: 'CUDA engine uninstalled — re-install it anytime with the button below.',
      wiz_uninstall_busy: 'The download wizard is running — wait for it to finish before uninstalling.',
      wiz_uninstall_in_use: 'A synthesis is using the CUDA engine — try again once it finishes.',
      wiz_uninstall_not_pack: 'That folder does not look like this plugin’s accelerator pack — nothing was deleted.',
      wiz_uninstall_io: 'Uninstall failed (file locked or permission denied) — try again shortly.',
      wiz_gpu_title: 'GPU accelerator pack (optional)',
      wiz_gpu_on: 'Installed · CUDA acceleration active',
      wiz_gpu_off: 'Not installed · CPU voice works; this makes it much faster.',
      wiz_gpu_nvidia: 'Acceleration needs an NVIDIA GPU; AMD / integrated-GPU users keep the default CPU mode — speech works the same.',
      wiz_gpu_btn: (gb) => `Download GPU pack (~${gb})`,
      wiz_gpu_note: '~750 MB from GitHub (fastest source auto-picked, mirror fallback), auto-extracted and verified; acceleration applies on the next utterance.',
      wiz_gpu_locked: 'Download or load a model first — this unlocks automatically',
      path_who: 'your-username',
      stt_env_ready_desc: (py) => `verified loadable engine environment: ${py}`,
      stt_env_missing_desc: 'The engine environment is not installed yet. It is a private Python venv the plugin owns, inside the Hermes data directory — official Hermes upgrades never touch it. One click builds it.',
      stt_env_deps_desc: 'The environment is there, but its transcription engine is gone — an environment rebuild most likely wiped it (these engines are not in Hermes\u2019 dependency lock). One click puts it back; downloaded models stay.',
      stt_env_broken_desc: 'The folder is there, but its Python no longer starts (the host interpreter was replaced, or a previous install was interrupted). One click rebuilds it.',
      stt_env_install: 'Install ASR environment',
      stt_env_rebuild: 'Reinstall environment',
      stt_env_wipe: 'clearing the old environment',
      stt_env_venv: 'creating the venv',
      stt_env_pip: 'installing the engine',
      stt_env_verify: 'verifying the engine loads',
      stt_env_prog: (s, pct, el) => `${s} · ${pct}% · ${el}s elapsed`,
      stt_env_done: 'ASR environment is ready',
      stt_env_fail: 'environment install failed: ',
      stt_env_unfail: 'environment uninstall failed: ',
      open_log: 'Open log',
      open_log_fail: 'could not open the log: ',
      stt_env_uninstall: 'Uninstall environment',
      stt_env_uninstall_confirm: 'Click again to confirm uninstall',
      stt_env_uninstalled: 'ASR environment removed — the model cache stays; delete it separately in the model row if you like.',
      stt_env_disk: (gb) => `Heads-up: only ${gb} GB free on the target disk — environment plus models need at least 1.5 GB.`,
      stt_env_other: (py) => `Note: the interpreter actually in use is an outside Python: ${py}`,
      stt_retry: 'Re-check',
      stt_model: 'Model',
      stt_cached: 'already cached',
      stt_dl: 'not cached — download available',
      stt_dir_entry: 'Local model directory…',
      stt_dir_label: 'Local model directory (must be CTranslate2: model.bin / tokenizer.json present)',
      stt_dir_browse: 'Browse…',
      stt_device: 'Compute device',
      stt_dev_auto: 'Auto (prefers GPU)',
      stt_lang: 'Language',
      stt_lang_auto: 'Auto-detect',
      stt_save: 'Verify & save',
      stt_save_locked: 'You picked a model that is not on this machine yet: press "Download this model" above first — this button unlocks once the file is fully local.',
      stt_dl_btn: (gb) => `Download this model (~${gb} GB, may take minutes)`,
      stt_dl_busy: 'Downloading — do not close the window…',
      stt_dl_hint: (gb) => `This model is not on this machine yet (~${gb} GB): press "Download this model" first — download failure = network/proxy, save failure = engine; two honest steps.`,
      stt_dl_ok: (m, s) => `Downloaded and really loaded: ${m} is now in the local cache (loaded in ${s}s) — "Verify & save" is unlocked.`,
      stt_dl_cached: (m) => `${m} is already in the local cache — just press "Verify & save".`,
      stt_dl_fail: 'Download failed (network/proxy?): ',
      stt_dl_done: (m) => `Downloaded: ${m} is now in the local cache — "Verify & save" just unlocked.`,
      stt_dl_prog: (p, r, e) => `${p}% · ${r} MB/s · ETA ${e}`,
      stt_dl_eta_calc: 'calculating ETA…',
      stt_del_btn: (gb) => `Delete this model (frees ~${gb} GB)`,
      stt_del_active: (m) => `${m} is the active engine — switch to another model and save before deleting it`,
      stt_del_title: 'Delete speech-to-text model',
      stt_del_desc: (m, gb) => `This removes the local cache of ${m} (~${gb} GB) from disk for good. Want it back? "Download this model" fetches it again — nothing is kept in between.`,
      stt_del_confirm: 'Delete',
      stt_del_busy: 'Deleting…',
      stt_del_done: 'Deleted',
      stt_del_ok: (m, gb) => `Deleted ${m} — freed ~${gb} GB.`,
      stt_verifying: 'Engine is loading from the local cache… (a few seconds)',
      stt_saved: (m, d, s) => `Saved & verified: ${m} · device ${d} · loaded in ${s}s. Live from the next utterance.`,
      stt_rejected: 'Save rejected (old settings kept): ',
      stt_need_model: 'Pick a model size first, or fill a local model directory',
      stt_now: (m, d, l) => `Live now: ${m || '—'} · device ${d} · language ${l}`,
      soon_clone: 'Voice management and “New clone” (reference audio + text, saved only after a real synthesis check) arrive next.',
      clone_new: 'New clone',
      voice_unset: 'no voice set',
      models_missing: (list) => 'Missing model files: ' + list + ' (download links in the plugin README)',
      tokenizer_missing: 'Tokenizer missing (models/tokenizer.json)',
      mdl_title: 'Model directory',
      mdl_hint: 'Tell Yache where the models live: Pick the folder containing qwen-talker-1.7b-*.gguf (the parent folder or the models/ subfolder both work). "Read" really reads the folder and only applies on success; the engine re-reads this per utterance, so switching is instant — no restart needed.',
      mdl_ph: 'Model folder path (pasting is fine too)',
      mdl_fresh: 'No model yet — use the download wizard above and pick any folder you like (no need to keep it on C:); it auto-loads here when done. Already have models somewhere? Pick that folder and hit "Read" to load it.',
      mdl_pick: 'Choose',
      mdl_read: 'Read',
      mdl_reset: 'Use default',
      mdl_ok: (n) => 'Directory applied: ' + n + ' model variant(s) found',
      mdl_reset_ok: 'Back to the default directory',
      mdl_dirty: 'Directory path changed but not confirmed — press \u201cRead\u201d to scan it before the models are listed',
      vt_title: 'Model variant',
      vt_ph: 'Pick a variant first',
      vt_base: 'Base — clone a real voice from reference audio',
      vt_cv: 'CustomVoice — pick a built-in voice',
      vt_vd: 'VoiceDesign — describe a voice in words',
      vt_hint: "Point at the model folder and press \"Read\"; the parameter menus appear only after the scan identifies the variants.",
      vt_pick_hint: 'Choose a model variant above first',
      scan_err: (e) => 'Current model directory unusable: ' + e,
      grp_spk: (n) => 'Built-in · ' + n,
      grp_cln: (n) => 'Cloned · ' + n,
      grp_design: 'Designed voice (text-to-voice)',
      pick_voice: 'Voice',
      design_desc: 'Voice description',
      design_ph: 'e.g. a gentle young woman in her early 20s, warm smile, slow pace',
      voice_save: 'Save voice',
      voice_saving: 'Saving… (real synthesis check)',
      pick_applied_clone: (n) => `\u2705 Selected the ${n} clone voice \u2014 TTS synthesis will play with it!`,
      pick_applied_spk: (n) => `\u2705 Selected the ${n} built-in voice \u2014 TTS synthesis will play with it!`,
      clone_next_step: 'Next step: go back to the overview, open the settings of the TTS Speech Synthesis section, pick a synthesis tier and save — the assistant will speak with this voice.',
      voice_saved: 'Voice saved as current ✓ TTS synthesis unlocked',
      dz_export: 'Export clip',
      dz_export_tip: 'Each design run invents a new random voice — exporting is how you keep the one you just heard',
      dz_adopt: 'Adopt as clone voice',
      dz_adopt_tip: 'Name it and the voice you just heard becomes a fixed clone voice used for every future read (names: letters, digits, _ only)',
      dz_name_ph: 'Name this voice first',
      dz_need_name: 'Please give this voice a name first.',
      dz_need_clip: 'Press Preview first — there is nothing to export yet.',
      dz_ready: 'This preview is ready: export it, or name it and adopt it as a clone voice.',
      dz_saved: 'Exported: ',
      dz_adopted: 'Clone voice registered and set as current: ',
      dz_drop: 'Remove this clone voice',
      dz_drop_confirm: 'Click again to confirm',
      dz_drop_tip: 'Drops the registry entry only; the wav stays on disk for you to delete yourself',
      dz_dropped: 'Removed: ',
      dz_dropped_wav: ' (the recording stays on disk)',
      voice_fail: 'Failed: ',
      aud_ok: (s, e) => 'Audition done: ' + s + 's of audio in ' + e + 's',
      clone_new_title: 'New cloned voice',
      clone_name_ph: 'voice name (letters, digits, _ or CJK)',
      clone_ref_ph: 'reference audio path (wav/mp3; a bare filename resolves inside the voice library)',
      clone_ref_pick: 'Pick…',
      clone_text_ph: 'what the reference says (optional, auto-detected)',
      clone_desc_ph: 'description (optional)',
      clone_go: 'Create & verify',
      ref_asr_busy: 'Transcribing the reference audio, please wait…',
      ref_asr_ok: 'Transcript filled in below — verify it, edits are welcome.',
      ref_asr_fail: 'Could not transcribe it. Type what the audio says below.',
      clone_none: 'No clone yet — press "New clone" below, pick a reference audio file from your machine, the words get auto-detected, verify and save to use it',
      q_auto: 'Smart (auto by free VRAM)',
      q_q8_0: 'Best quality · Q8_0',
      q_q4_k_m: 'VRAM saver · Q4_K_M',
      q_bf16: 'Full precision · BF16',
      q_missing: 'missing',
      d_auto: 'Auto (GPU first)',
      d_cuda: 'GPU (CUDA)',
      d_cpu: 'CPU',
      tts_quant: 'Quantization',
      tts_device: 'Compute backend',
      tts_save: 'Save synthesis config',
      tts_saving: 'Verifying… (one real synthesis at the new tier)',
      tts_dirty_tip: 'Changes not saved yet — press the button on the left to save and verify.',
      verify_failed: '❌ Verification failed:',
      verify_unknown: 'the backend gave no reason (check the plugin log).',
      tts_saved: 'Saved — the next reply uses it ✓',
      cur_voice: 'Current voice',
      rec_tier: 'Recommended (voice + free VRAM)',
      cur_backend: 'Engine actually uses',
      tts_free_vram: 'Free VRAM',
    },
  },
};

function navigate(path) {
  try { if (host && typeof host.navigate === 'function') return host.navigate(path); } catch { /* noop */ }
}
function goBack() {
  try {
    if (window.history.length > 1) window.history.back();
    else navigate('/');
  } catch { navigate('/'); }
}

const LAMP_IDS = ['plugin', 'engine', 'models', 'vram', 'voice', 'worker'];

function useDoctor(ctx, active) {
  const [doc, setDoc] = useState(null);
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');
  const refresh = useCallback(() => {
    const d = ctx.rest('/doctor', { timeoutMs: 25000 })
      .then((x) => { setDoc(x); setErr(''); })
      .catch((e) => setErr(String(e?.message || e)));
    const s = ctx.rest('/status', { timeoutMs: 12000 })
      .then((x) => setSt(x)).catch(() => {});
    return Promise.all([d, s]).then(() => {});
  }, [ctx]);
  useEffect(() => {
    if (!active) return;
    refresh();
    const h = setInterval(refresh, 30000);
    return () => clearInterval(h);
  }, [active, refresh]);
  return { doc, st, err, refresh };
}

/* The voice library view: on-disk scan + saved selection + synthesis
   config, all from one backend route. The clone -> TTS unlock reads
   lib.saved_voice — the panel-saved truth — never a guess. */
function useVoiceLibrary(ctx, active) {
  const [lib, setLib] = useState(null);
  const [err, setErr] = useState('');
  const refresh = useCallback(() => {
    ctx.rest('/voice/library', { timeoutMs: 30000 })
      .then((d) => { setLib(d); setErr(''); })
      .catch((e) => setErr(String((e && e.message) || e)));
  }, [ctx]);
  useEffect(() => { if (active) refresh(); }, [active, refresh]);
  return { lib, err, refresh };
}

function voiceView(v, t) {
  if (!v || !v.kind) return '';
  const sz = String(v.model_size || '').trim().toLowerCase() === '0.6b'
    ? ' · 0.6B' : (v.model_size ? ' · 1.7B' : '');
  if (v.kind === 'speaker') return t('yache.grp_spk', String(v.speaker || '?')) + sz;
  if (v.kind === 'clone') return t('yache.grp_cln', String(v.voice || '?')) + sz;
  return t('yache.grp_design') + sz;
}

const QUANT_KEYS = { auto: 'yache.q_auto', q8_0: 'yache.q_q8_0',
                     q4_k_m: 'yache.q_q4_k_m', bf16: 'yache.q_bf16' };
const DEVICE_KEYS = { auto: 'yache.d_auto', cuda: 'yache.d_cuda',
                      cpu: 'yache.d_cpu' };
const quantLabel = (q, t) => (QUANT_KEYS[q] ? t(QUANT_KEYS[q]) : String(q).toUpperCase());
const deviceLabel = (d, t) => t(DEVICE_KEYS[d] || 'yache.d_auto');

/* One-shot player for the base64 wav that preview/save/clone routes
   return — hearing is the whole point of this panel. */
function playWav(b64) {
  if (!b64) return;
  try { new Audio('data:audio/wav;base64,' + b64).play(); }
  catch { /* autoplay policy may block; the note text already tells the result */ }
}

/* Activation lights — the honest answer to "who is actually serving
   right now?".  A provider key is one scalar in the official config; when
   it points elsewhere our engines sleep (zero RAM/VRAM), the official
   provider entry keeps full control and we never block it.  The panel
   then says "ready · dormant" instead of pretending to serve. */
const STATE_DOT = { serving: 'bg-emerald-500', ready: 'bg-sky-500', notready: 'bg-amber-500' };

function useActivation(ctx, active) {
  const [act, setAct] = useState(null);
  const [busy, setBusy] = useState(null);
  const refresh = useCallback(() => {
    if (!ctx || !active) return Promise.resolve();
    return ctx.rest('/activation', { timeoutMs: 8000 })
      .then((d) => { if (d) setAct(d); }).catch(() => {});
  }, [ctx, active]);
  useEffect(() => {
    refresh();
    if (!active) return undefined;
    const timer = setInterval(refresh, 30000);
    return () => clearInterval(timer);
  }, [refresh, active]);
  /* One click back to local: the backend probes the real engine first, so
     a cold machine never wakes up silent after the click.  Model downloads
     can make this slow — 4 minutes, and the button label says so. */
  const switchTo = useCallback(async (module) => {
    setBusy(module);
    try {
      const d = await ctx.rest('/activation', {
        method: 'POST', timeoutMs: 240000, json: { module, probe: true },
      });
      if (d && d.ok) {
        setAct(d.activation || null);
        try { host.notify({ kind: 'info', message: t(module === 'stt' ? 'yache.switch_ok_stt' : 'yache.switch_ok') }); } catch { /* best-effort toast */ }
        return true;
      }
      try { host.notify({ kind: 'warning', message: t('yache.switch_failed') + ': ' + ((d && d.error) || '') }); } catch { /* noop */ }
      if (d && d.activation) setAct(d.activation);
      return false;
    } catch (e) {
      try { host.notify({ kind: 'warning', message: t('yache.switch_failed') + ': ' + String((e && e.message) || e) }); } catch { /* noop */ }
      return false;
    } finally { setBusy(null); }
  }, [ctx]);
  return { act, busy, switchTo, refresh };
}
function LampRow({ t, lamp, full }) {
  const ok = lamp && lamp.ok;
  const dot = ok ? 'bg-emerald-500' : 'bg-rose-500';
  return jsxs('div', { className: 'flex items-start gap-2 py-0.5', children: [
    jsx('span', { className: cn('mt-1 size-2 shrink-0 rounded-full', dot) }),
    jsxs('div', { className: 'min-w-0 flex-1', children: [
      jsx('div', { className: 'text-xs font-medium', children: t('yache.lamp.' + lamp.id) }),
      full && lamp.detail
        ? jsx('div', { className: 'truncate text-[10px] text-muted-foreground',
                       title: maskPath(lamp.detail, t('yache.path_who')), children: maskPath(lamp.detail, t('yache.path_who')) })
        : null,
    ] }),
  ] });
}

function StopPhrase({ ctx, t }) {
  const [cur, setCur] = useState(null);
  const [phrase, setPhrase] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');

  useEffect(() => {
    let live = true;
    ctx.rest('/stop-phrase', { timeoutMs: 8000 })
      .then((d) => {
        if (!live || !d) return;
        setCur(d);
        if (Array.isArray(d.phrases) && d.phrases.length) setPhrase(d.phrases.join(', '));
      })
      .catch(() => {});
    return () => { live = false; };
  }, [ctx]);

  const save = useCallback(async () => {
    setBusy(true);
    setNote('');
    try {
      const r = await ctx.rest('/stop-phrase', { method: 'POST',
        body: { phrase: phrase.trim() }, timeoutMs: 15000 });
      if (r && Array.isArray(r.phrases)) {
        setPhrase(r.phrases.join(', '));
        setCur({ configured: true, phrases: r.phrases,
                 default_phrases: (cur && cur.default_phrases) || [] });
      }
      setNote(t('yache.stop_saved'));
      // Same hot-apply as an official settings save: the desktop config
      // query (['hermes-config-record']) refetches, so the host re-reads
      // the phrase table immediately — no restart needed.
      try { queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] }); }
      catch { /* noop */ }
    } catch (err) {
      setNote(t('yache.stop_fail') + ': ' + String((err && err.message) || err));
    } finally { setBusy(false); }
  }, [ctx, cur, phrase, t]);

  return jsxs('div', { children: [
    jsxs('div', { className: 'mb-1 flex items-center justify-between', children: [
      jsx('div', { className: 'text-xs font-semibold', children: t('yache.stop_title') }),
      cur && cur.configured ? null
        : jsx('span', { className: 'text-[10px] text-muted-foreground',
            title: (cur && Array.isArray(cur.default_phrases) && cur.default_phrases.join(', ')) || '',
            children: t('yache.stop_default') }),
    ] }),
    jsx('p', { className: 'mb-2 text-[11px] leading-relaxed text-muted-foreground',
      children: t('yache.stop_desc') }),
    jsxs('div', { className: 'flex items-center gap-2', children: [
      jsx(Input, { value: phrase, placeholder: t('yache.stop_ph'),
        className: 'flex-1 h-8 text-xs',
        onChange: (e) => setPhrase(e.target.value),
        onKeyDown: (e) => { if (e.key === 'Enter' && !e.nativeEvent.isComposing && phrase.trim() && !busy) save(); } }),
      jsx(Button, { size: 'xs', disabled: busy || !phrase.trim(), onClick: save,
        children: busy ? t('yache.stop_saving') : t('yache.stop_save') }),
    ] }),
    note ? jsx('p', { className: 'mt-1 text-[11px] '
      + (note.startsWith(t('yache.stop_fail')) ? 'text-rose-500' : 'text-muted-foreground'),
      children: note }) : null,
  ] });
}

/* Wake-up conversation mode. The stock wake flow re-reads this switch each
   time it arms a conversation, so we render the truth: mode = fresh window
   per wake, keep = stay in the current conversation. Reads and writes go
   through the same official config channel as the stop-phrase card; saving
   does NOT open a session on its own, it only flips the stored mode. */
function WakeSession({ ctx, t }) {
  const [cur, setCur] = useState(null);
  const [mode, setMode] = useState(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');

  useEffect(() => {
    let live = true;
    ctx.rest('/wake-session', { timeoutMs: 8000 })
      .then((d) => { if (!live || !d) return; setCur(d);
        setMode(d.start_new_session ? 'new' : 'keep'); })
      .catch(() => {});
    return () => { live = false; };
  }, [ctx]);

  const dirty = !!(cur && mode && mode !== (cur.start_new_session ? 'new' : 'keep'));

  const save = useCallback(async () => {
    if (!mode) return;
    setBusy(true);
    setNote('');
    try {
      const r = await ctx.rest('/wake-session', { method: 'POST',
        body: { start_new_session: mode === 'new' }, timeoutMs: 15000 });
      if (r && typeof r.start_new_session === 'boolean') setCur(r);
      setNote(t('yache.wake_saved'));
      /* Keep every official config consumer in sync through the shared cache. */
      try { queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] }); }
      catch { /* noop */ }
    } catch (err) {
      setNote(t('yache.wake_fail') + ': ' + String((err && err.message) || err));
    } finally { setBusy(false); }
  }, [ctx, mode, t]);

  return jsxs('div', { children: [
    jsxs('div', { className: 'mb-1 flex items-center justify-between', children: [
      jsx('div', { className: 'text-xs font-semibold', children: t('yache.wake_title') }),
      cur && cur.configured ? null
        : jsx('span', { className: 'text-[10px] text-muted-foreground',
            children: t('yache.wake_default') }),
    ] }),
    jsx('p', { className: 'mb-2 text-[11px] leading-relaxed text-muted-foreground',
      children: t('yache.wake_desc') }),
    jsxs('div', { className: 'flex items-center gap-2', children: [
      jsxs(Select, {
        value: mode || 'new',
        onValueChange: setMode,
        disabled: busy || !mode,
        children: [
          jsx(SelectTrigger, { className: 'h-8 flex-1 text-xs', children: jsx(SelectValue, {}) }),
          jsx(SelectContent, { children: [
            jsx(SelectItem, { value: 'new', children: t('yache.wake_opt_new') }, 'new'),
            jsx(SelectItem, { value: 'keep', children: t('yache.wake_opt_keep') }, 'keep'),
          ] }),
        ] }),
      jsx(Button, { size: 'xs', disabled: busy || !dirty, onClick: save,
        children: busy ? t('yache.wake_saving') : t('yache.wake_save') }),
    ] }),
    note ? jsx('p', { className: 'mt-1 text-[11px] '
      + (note.startsWith(t('yache.wake_fail')) ? 'text-rose-500' : 'text-muted-foreground'),
      children: note }) : null,
  ] });
}

function QuickPanel({ ctx, t, close }) {
  const { doc, st, err, refresh } = useDoctor(ctx, true);
  const { act, busy: actBusy, switchTo: actSwitch, refresh: refreshAct } = useActivation(ctx, true);
  const providerOfQuick = (a, k) => { const p = a && a.providers && a.providers[k]; return (!p || p === '(stock default)') ? '' : p; };
  const dormantQuick = (a, k) => { const m = a && a.modules && a.modules[k]; return !!(providerOfQuick(a, k) && m && m.state === 'ready'); };
  const lamps = (doc && doc.lamps) || LAMP_IDS.map((id) => ({ id, ok: false }));
  return jsxs('div', { className: 'flex flex-col gap-1 p-1 w-[360px]', children: [
    jsxs('div', { className: 'flex items-center justify-between px-1', children: [
      jsxs('div', { className: 'flex items-center gap-1.5 text-sm font-semibold', children: [
        jsx(Codicon, { name: 'comment-discussion', size: 16 }),
        jsx('span', { children: t('yache.brand') }),
      ] }),
      jsxs('div', { className: 'flex items-center gap-1', children: [
        jsx(Tip, { content: t('yache.refresh'), children:
          jsx(Button, { variant: 'ghost', size: 'xs',
                        onClick: () => { refresh(); refreshAct(); },
                        title: t('yache.refresh'),
                        children: jsx(icons.RefreshCw || icons.Power || 'span',
                                      { className: 'size-3.5' }) }) }),
        jsx(Button, { variant: 'outline', size: 'xs',
                      onClick: () => { close(); navigate(pluginSettingsHref(ID)); },
                      children: t('yache.settings') }),
      ] }),
    ] }),
    err
      ? jsx('div', { className: 'px-1 text-[10px] text-rose-500', children: err })
      : null,
    jsx('div', { className: 'text-[10px] font-medium text-muted-foreground px-1 pt-1',
                 children: t('yache.doctor') }),
    jsx(Fragment, { children: lamps.map((l) =>
      jsx(LampRow, { t, lamp: l }, l.id)) }),
    act ? jsxs('div', { className: 'flex justify-between gap-3 border-t border-border py-1 text-xs', children: [
      jsxs('span', { className: 'inline-flex shrink-0 items-center gap-1.5', children: [
        jsx('span', { className: cn('size-1.5 rounded-full', STATE_DOT[act.state] || 'bg-muted-foreground/30') }),
        jsx('span', { children: t('yache.status_' + act.state) }),
      ] }),
      jsxs('span', { className: 'inline-flex min-w-0 items-center gap-1.5', children: [
        jsx('span', { className: 'max-w-[8.5em] truncate text-[10px] text-muted-foreground',
                      title: t('yache.badge_tts') + ' · ' + t('yache.status_' + (((act.modules || {}).tts || {}).state || 'notready'))
                           + ' · ' + t('yache.badge_stt') + ' · ' + t('yache.status_' + (((act.modules || {}).stt || {}).state || 'notready'))
                           + ' · ' + t('yache.badge_wake') + ' · ' + t('yache.status_' + (((act.modules || {}).wake || {}).state || 'notready')),
                      children: t('yache.badge_tts') + ' ' + t('yache.status_' + (((act.modules || {}).tts || {}).state || 'notready')) }),
        dormantQuick(act, 'tts') ? jsx(Button, { size: 'xs', variant: 'outline', disabled: !!actBusy,
          title: t('yache.switching'), onClick: () => actSwitch('tts'),
          children: actBusy === 'tts' ? t('yache.switching') : t('yache.switch_back') }) : null,
      ] }),
    ] }) : null,
    act && dormantQuick(act, 'stt') ? jsxs('div', { className: 'flex items-center justify-between gap-3 py-0.5 text-xs', children: [
      jsxs('span', { className: 'inline-flex shrink-0 items-center gap-1.5', children: [
        jsx('span', { className: 'size-1.5 rounded-full bg-sky-500' }),
        jsx('span', { children: t('yache.badge_dormant') + ' ' + t('yache.badge_stt') }),
      ] }),
      jsxs('span', { className: 'inline-flex min-w-0 items-center gap-1.5', children: [
        jsx('span', { className: 'max-w-[8.5em] truncate text-[10px] text-muted-foreground',
                      title: providerOfQuick(act, 'stt'), children: providerOfQuick(act, 'stt') }),
        jsx(Button, { size: 'xs', variant: 'outline', disabled: !!actBusy,
          title: t('yache.switching'), onClick: () => actSwitch('stt'),
          children: actBusy === 'stt' ? t('yache.switching') : t('yache.switch_back') }),
      ] }),
    ] }) : null,
    st ? jsxs(Fragment, { children: [
      jsxs('div', { className: 'flex items-center justify-between px-1 pt-1 text-[11px]', children: [
        jsx('span', { className: 'text-muted-foreground',
                      children: st.has_nvidia ? t('yache.vram') : t('yache.vram_none') }),
        jsx('span', { className: 'font-mono',
                      children: st.has_nvidia && st.free_vram_mib != null
                        ? st.free_vram_mib + ' MiB' : '—' }),
      ] }),
      st.recommended ? jsxs('div', { className: 'flex items-center justify-between px-1 text-[11px]', children: [
        jsx('span', { className: 'text-muted-foreground', children: t('yache.rec') }),
        jsx('span', { className: 'font-mono', children: st.recommended.tier }),
      ] }) : null,
    ] }) : null,
    jsx('div', { className: 'flex justify-end pt-1 border-t border-border', children:
      jsx(Button, { size: 'xs', variant: 'outline', title: t('yache.done_title'),
                    onClick: close, children: t('yache.done') }) }),
  ] });
}

// Declarative status-bar chip (same shape as hermes-wallpaper): the host
// renders the official StatusBar trigger, so native hover styling and the
// right-click "show in status bar" checklist both apply automatically.
function ChipIcon({ ctx }) {
  const { doc } = useDoctor(ctx, true);
  const healthy = doc ? doc.ok : null;
  const dot = healthy === null ? 'bg-muted-foreground/50'
            : healthy ? 'bg-emerald-500' : 'bg-rose-500';
  return jsx('span', {
    className: 'relative inline-flex h-4 w-4 items-center justify-center',
    'aria-hidden': true,
    children: [
      jsx(Codicon, { name: 'comment-discussion', size: 14, className: 'text-muted-foreground' }),
      jsx('span', {
        className: `absolute -right-1 -top-1 h-1.5 w-1.5 rounded-full ring-1 ring-background ${dot}`,
      }),
    ],
  });
}
function PanelMenu({ ctx, close }) {
  const t = usePluginI18n(ID);
  return jsx(QuickPanel, { ctx, t, close });
}
function VoiceChip({ ctx }) {
  const t = usePluginI18n(ID);
  const [open, setOpen] = useState(false);
  const { doc } = useDoctor(ctx, open);
  const healthy = doc ? doc.ok : null;
  const dot = healthy === null ? 'bg-muted-foreground/50'
            : healthy ? 'bg-emerald-500' : 'bg-rose-500';
  return jsx(Popover, { open, onOpenChange: setOpen, children: jsxs(Fragment, { children: [
    jsx(PopoverTrigger, { asChild: true, children:
      jsx(RowButton, { tone: 'chrome',
        title: t('yache.tip') + (healthy === null ? '' : ' · ' + (healthy ? 'OK' : 'FAIL')),
        children: jsxs('span', { className: 'flex items-center gap-1.5', children: [
          jsx(Codicon, { name: 'comment-discussion', size: 14 }),
          jsx('span', { className: 'text-[11px]', children: t('yache.brand') }),
          jsx('span', { className: cn('size-1.5 rounded-full', dot) }),
        ] }) }) }),
    jsx(PopoverContent, { align: 'end', side: 'top', className: 'w-auto p-1', children:
      open ? jsx(QuickPanel, { ctx, t, close: () => setOpen(false) }) : null }),
  ] }) });
}

function StatusRow({ label, value, title }) {
  return jsxs('div', { className: 'flex justify-between gap-3 py-0.5 text-xs', children: [
    jsx('span', { className: 'shrink-0 text-muted-foreground',
                  title: title || label, children: label }),
    jsx('span', { className: 'max-w-[70%] truncate text-right font-mono',
                  title: title || value || '', children: value }),
  ] });
}

/* Modal shell per the locked spec: centered, ~640px (never fullscreen),
   internal scrolling, explicit Back (Esc / backdrop also exit \u2014 explicit
   exits everywhere, no dead ends). Save-with-self-check buttons arrive
   with each section's parameters next round. */
function ModalShell({ t, title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return jsx('div', {
    className: 'fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4',
    onClick: onClose,
    children: jsx('div', {
      className: 'flex max-h-[85vh] w-[640px] max-w-full flex-col overflow-hidden rounded-xl border border-border bg-background shadow-2xl',
      onClick: (e) => e.stopPropagation(),
      children: [
        jsx('div', { className: 'flex shrink-0 items-center justify-between border-b border-border px-4 py-2.5', children: [
          jsx('div', { className: 'text-sm font-semibold', children: title }),
          jsx(Button, { size: 'xs', variant: 'outline', onClick: onClose,
                        children: t('yache.back') }),
        ] }),
        jsx('div', { className: 'min-h-0 flex-1 overflow-y-auto p-4', children }),
      ],
    }),
  });
}

/* Synthesis config: quant tier + backend, saved through a REAL synthesis
   probe under the new config. Locked until a voice is saved in the cloning
   section — hearing what a tier sounds like only makes sense on a chosen
   voice, and the tier/VRAM pairing stays in one human loop. */
function TtsBody({ ctx, t, st, lib, refreshLib }) {
  const [quant, setQuant] = useState('auto');
  const [device, setDevice] = useState('auto');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const [noteOk, setNoteOk] = useState(null);
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current || !lib || !lib.saved_tts) return;
    seeded.current = true;
    setQuant(String(lib.saved_tts.quant || 'auto'));
    setDevice(String(lib.saved_tts.device || 'auto'));
  }, [lib]);
  const dirty = !!lib && (quant !== String((lib.saved_tts || {}).quant || 'auto')
    || device !== String((lib.saved_tts || {}).device || 'auto'));
  const rec = lib && lib.recommend;
  const missing = lib ? Object.entries(lib.scan || {})
    .filter(([, v]) => !(v && v.present)).map(([k]) => k) : [];
  /* Quant menu is built LIVE from the scan (files on disk) for the combo
     the saved voice actually uses — variant × model size. The same Q8_0
     label on the 1.7b and 0.6b builds costs very different VRAM, so the
     menu follows the pinned size and every tier line carries the real
     file size of THIS build. A saved-but-now-missing tier stays selectable
     (labelled) so the select never renders blank. */
  const _av = (lib && lib.saved_voice && lib.saved_voice.variant)
    || (lib && lib.anchor_variant) || 'base';
  const _asz = String((lib && lib.anchor_size)
    || (lib && lib.saved_voice && lib.saved_voice.model_size) || '')
    .trim().toLowerCase();
  const _rows = ((((lib && lib.scan) || {})[_av] || {}).quants || [])
    .map((x) => (x && typeof x === 'object'
      ? x : { quant: String(x), model_size: '' }));
  const _sizeKey = (x) => String(x.model_size || '').trim()
    .toLowerCase() === '0.6b' ? '0.6b' : '1.7b';
  const _sized = _asz ? _rows.filter((x) => _sizeKey(x) === _asz) : _rows;
  const _menu = _sized.length ? _sized : _rows;
  const availQuants = (() => {
    const seen = new Set();
    _menu.forEach((x) => { const q = String(x.quant || '').toLowerCase(); if (q) seen.add(q); });
    return Array.from(seen);
  })();
  const quantOpts = (() => {
    const seen = new Set(availQuants);
    // keep a saved-but-missing tier selectable so the select never blanks
    if (quant !== 'auto' && !seen.has(quant)) seen.add(quant);
    if (quant === 'auto') seen.add('auto');
    return Array.from(seen).sort();
  })();
  const tierMib = (() => {
    const m = {};
    _menu.forEach((x) => {
      const q = String(x.quant || '').toLowerCase();
      const n = Number(x.size_mib);
      if (q && n > 0 && m[q] == null) m[q] = n;
    });
    return m;
  })();
  const fmtMib = (mib) => mib >= 1024
    ? `${Math.round(mib / 102.4) / 10} G` : `${Math.round(mib)} MB`;
  const qLabel = (q) => (q !== 'auto' && availQuants.indexOf(q) < 0)
    ? quantLabel(q, t) + ' · ' + t('yache.q_missing')
    : (q === 'auto' || tierMib[q] == null) ? quantLabel(q, t)
    : `${quantLabel(q, t)} (${fmtMib(tierMib[q])})`;
  const saveTTS = useCallback(async () => {
    setBusy(true); setNote(''); setNoteOk(null);
    try {
      const r = await ctx.rest('/voice/tts', { method: 'POST',
        body: { quant, device }, timeoutMs: 300000 });
      if (r && r.ok) {
        setNoteOk(true);
        setNote(t('yache.tts_saved'));
        playWav(r.wav_b64);
        if (refreshLib) refreshLib();
      } else {
        setNoteOk(false);
        setNote(t('yache.verify_failed') + ' '
          + String((r && r.error) || t('yache.verify_unknown')));
      }
    } catch (err) {
      setNoteOk(false);
      setNote(t('yache.verify_failed') + ' '
        + String((err && err.message) || err));
    } finally { setBusy(false); }
  }, [ctx, quant, device, t, refreshLib]);
  return jsxs('div', { className: 'space-y-2', children: [
    lib && lib.saved_voice ? StatusRow({ label: t('yache.cur_voice'),
      value: voiceView(lib.saved_voice, t) }) : null,
    st ? StatusRow({ label: t('yache.cur_backend'), value: st.selected || 'auto',
      title: (st.recommended && st.recommended.reason) || '' }) : null,
    rec ? StatusRow({ label: t('yache.rec_tier'),
      value: quantLabel(rec.quant, t) + (rec.model_size ? ` · ${String(rec.model_size).toUpperCase()}` : '') + ' · ' + deviceLabel(rec.device, t),
      title: rec.reason || '' }) : null,
    lib && lib.has_cuda ? StatusRow({ label: t('yache.tts_free_vram'),
      value: lib.free_vram != null ? lib.free_vram + ' MiB' : '—' }) : null,
    missing.length ? jsx('p', { className: 'text-[11px] text-amber-500',
      children: t('yache.models_missing', missing.join(', ')) }) : null,
    lib && !lib.tokenizer_ok ? jsx('p', { className: 'text-[11px] text-rose-500',
      children: t('yache.tokenizer_missing') }) : null,
    jsxs('div', { className: 'grid grid-cols-2 gap-2 pt-1', children: [
      jsxs('div', { children: [
        jsx('div', { className: 'mb-1 text-[11px] text-muted-foreground',
                     children: t('yache.tts_quant') }),
        jsxs(Select, { value: quant, onValueChange: setQuant, disabled: busy, children: [
          jsx(SelectTrigger, { className: 'h-8 w-full text-xs',
                               children: jsx(SelectValue, {}) }),
          jsx(SelectContent, { children: quantOpts.map((q) =>
            jsx(SelectItem, { value: q, children: qLabel(q) }, q)) }),
        ] }),
      ] }, 'q'),
      jsxs('div', { children: [
        jsx('div', { className: 'mb-1 text-[11px] text-muted-foreground',
                     children: t('yache.tts_device') }),
        jsxs(Select, { value: device, onValueChange: setDevice, disabled: busy, children: [
          jsx(SelectTrigger, { className: 'h-8 w-full text-xs',
                               children: jsx(SelectValue, {}) }),
          jsx(SelectContent, { children: Object.keys(DEVICE_KEYS).map((d) =>
            jsx(SelectItem, { value: d, children: deviceLabel(d, t) }, d)) }),
        ] }),
      ] }, 'd'),
    ] }),
    jsxs('div', { className: 'flex items-center gap-2', children: [
      jsx(Button, { size: 'xs', disabled: busy || !dirty, onClick: () => void saveTTS(),
        children: busy ? t('yache.tts_saving') : t('yache.tts_save') }),
      dirty && !busy ? jsx('span', { className: 'text-[10px] text-amber-400',
        children: t('yache.tts_dirty_tip') }) : null,
    ] }),
    note ? jsx('p', { className: 'whitespace-pre-wrap text-[11px] '
      + (noteOk === false ? 'text-rose-500'
        : noteOk === true ? 'text-emerald-500' : 'text-muted-foreground'),
      children: note }) : null,
  ] });
}

/* STT transcription section — one official path (faster-whisper), no
   mirror detours. Opening runs the engine health check first: a red
   result stops the user before they fill anything. The model picker
   merges the size whitelist with the HF cache truth, so every option is
   honest about "cached" vs "downloads on save"; the "local model
   directory" tail entry swaps in a path input (CTranslate2 format —
   fingerprint-checked server-side). Save = the backend really loads the
   chosen model (--verify) before persisting anything; the engine's own
   error comes back verbatim and the old settings stay untouched. */
const STT_LANGS = { zh: '\u4e2d\u6587 (zh)', en: 'English (en)',
  ja: '\u65e5\u672c\u8a9e (ja)', ko: '\ud55c\uad6d\uc5b4 (ko)',
  es: 'Espa\u00f1ol (es)', fr: 'Fran\u00e7ais (fr)',
  de: 'Deutsch (de)', ru: '\u0420\u0443\u0441\u0441\u043a\u0438\u0439 (ru)' };
/* The ASR environment: a Python venv the plugin owns under HERMES_HOME.
   faster-whisper cannot live in the gateway environment — an official
   dependency rebuild regenerates that venv from the stock lock and drops
   anything extra, which is exactly how local STT went silent once. This
   row is therefore also the installer: it reports what is VERIFIED on
   disk (interpreter answers + the engine really imports), and one click
   builds or repairs it, streaming the real pip log tail while it works.
   Network happens only under the button, like the model Download button. */
function EnvRow({ ctx, t, onReady }) {
  const [env, setEnv] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => ctx.rest('/stt/env', { timeoutMs: 130000 })
    .then((d) => { if (d) setEnv(d); }).catch(() => {}), [ctx]);
  useEffect(() => { load(); }, [load]);
  // Reopening the dialog while an install runs resumes the live bar.
  useEffect(() => {
    let live = true;
    ctx.rest('/stt/env/install/status', { timeoutMs: 8000})
      .then((r) => { if (live && r && r.state === 'running') { setBusy(true); setJob(r); } })
      .catch(() => {});
    return () => { live = false; };
  }, [ctx]);
  useEffect(() => {
    if (!busy) return undefined;
    let stop = false;
    const tick = async () => {
      if (stop) return;
      try {
        const r = await ctx.rest('/stt/env/install/status', { timeoutMs: 8000 });
        if (stop || !r) return;
        setJob(r);
        if (r.state === 'done') {
          setBusy(false);
          load();
          if (onReady) onReady();
          try {
            host.notify({ kind: 'success', message: t('yache.stt_env_done') });
          } catch { /* best-effort toast */ }
        } else if (r.state === 'error' || r.state === 'idle') {
          // 'idle' = the gateway restarted mid-install: stop the spinner
          // rather than wait for a job that no longer exists.
          setBusy(false);
          load();
        }
      } catch { /* transport blip: keep polling */ }
    };
    tick();
    const timer = setInterval(tick, 1500);
    return () => { stop = true; clearInterval(timer); };
  }, [busy, ctx, load, onReady, t]);

  const install = useCallback(async (rebuild) => {
    if (busy) return;
    setBusy(true); setJob(null);
    // A "busy" rejection from the server is NOT a failure — a worker is
    // live (double click, or another window started it). Adopt that live
    // job so the progress bar re-attaches instead of painting a scary red
    // "安装失败:正在安装中". Only a real no-worker answer counts as error.
    const failUnlessLive = async (msg) => {
      try {
        const live = await ctx.rest('/stt/env/install/status', { timeoutMs: 8000 });
        if (live && live.state === 'running') { setJob(live); return; }
      } catch (e) { /* probe failed — fall through to the honest error */ }
      setBusy(false);
      setJob({ state: 'error', error: msg });
      load();
    };
    try {
      const r = await ctx.rest('/stt/env/install', { method: 'POST',
        timeoutMs: 60000, body: { rebuild: !!rebuild } });
      if (!r || !r.ok) {
        await failUnlessLive(String((r && (r.error || r.detail)) || 'HTTP error'));
      }
    } catch (err) {
      await failUnlessLive(String((err && err.message) || err));
    }
  }, [ctx, busy, load]);

  // The loop closes both ways: an installed environment must be removable
  // from the same place it was installed. Two-step inline confirm (no
  // dialog — owner's standing rule), amber on the arming click. Removal
  // runs server-side against the install-time snapshot: the folder is
  // wholly ours (its runtime __pycache__ files are part of the install,
  // not user data), so the backend identifies it first, then takes the
  // whole tree down — model cache and everything else stay untouched.
  const [unarm, setUnarm] = useState(0);
  const unarmTimer = useRef(null);
  const uninstall = useCallback(async () => {
    if (unarm === 0) {
      setUnarm(1);
      clearTimeout(unarmTimer.current);
      unarmTimer.current = setTimeout(() => setUnarm(0), 3000);
      return;
    }
    clearTimeout(unarmTimer.current); setUnarm(0);
    if (busy) return;
    setBusy(true); setJob(null);
    try {
      const r = await ctx.rest('/stt/env/uninstall',
        { method: 'POST', timeoutMs: 120000 });
      if (r && r.ok) { setBusy(false); setJob(null); load(); }
      else {
        setBusy(false);
        setJob({ state: 'error', src: 'uninstall',
          error: String((r && (r.error || r.detail)) || 'failed') });
        load();
      }
    } catch (err) {
      setBusy(false);
      setJob({ state: 'error', src: 'uninstall',
        error: String((err && err.message) || err) });
    }
  }, [unarm, busy, ctx, load]);

  const step = (k) => (k === 'wipe' ? t('yache.stt_env_wipe')
    : k === 'venv' ? t('yache.stt_env_venv')
      : k === 'pip' ? t('yache.stt_env_pip')
        : k === 'verify' ? t('yache.stt_env_verify') : String(k || ''));
  const st = env ? String(env.state || '') : '';
  const dot = busy ? 'bg-sky-500'
    : st === 'ready' ? 'bg-emerald-500'
      : st === 'missing' || st === 'deps' ? 'bg-amber-500' : 'bg-rose-500';
  const tail = ((job && job.log) || []).slice(-4).join('\n');
  const failed = !busy && job && job.state === 'error';
  return jsxs('div', { className: 'space-y-1.5 rounded-md border border-border/60 bg-muted/30 p-2', children: [
    jsxs('div', { className: 'flex items-center gap-2', children: [
      jsx('span', { className: cn('h-2 w-2 shrink-0 rounded-full', dot) }),
      jsx('span', { className: 'shrink-0 text-xs font-medium',
                    children: t('yache.stt_env_title') }),
      jsx('span', { className: 'truncate text-[11px] text-muted-foreground',
        children: !env ? t('yache.stt_env_check')
          : st === 'ready' ? t('yache.stt_env_ready')
            : st === 'missing' ? t('yache.stt_env_missing')
              : st === 'deps' ? t('yache.stt_env_deps')
                : t('yache.stt_env_broken') }),
      env && env.whisper_version && !busy ? jsx('span', {
        className: 'ml-auto shrink-0 text-[10px] text-muted-foreground',
        children: 'faster-whisper ' + String(env.whisper_version) }) : null,
    ] }),
    busy ? jsxs('div', { className: 'space-y-1', children: [
      jsx('div', { className: 'h-1.5 w-full overflow-hidden rounded-full bg-muted',
        children: jsx('div', {
          className: 'h-full rounded-full bg-sky-500 transition-all duration-700',
          style: { width: `${Math.max(3, Number(job && job.progress) || 0)}%` } }) }),
      jsx('p', { className: 'text-[10px] tabular-nums text-muted-foreground',
        children: t('yache.stt_env_prog',
          step((job && job.label) || ''),
          String(Math.round(Number(job && job.progress) || 0)),
          String(Math.round(Number(job && job.elapsed_s) || 0))) }),
    ] }) : null,
    !busy && env ? jsx('p', {
      className: 'text-[11px] leading-relaxed text-muted-foreground',
      children: st === 'ready'
        ? t('yache.stt_env_ready_desc', maskPath(env.env_python || '', t('yache.path_who')))
        : st === 'deps' ? t('yache.stt_env_deps_desc')
          : st === 'broken' ? t('yache.stt_env_broken_desc')
            : (env.detail ? String(env.detail)
              : t('yache.stt_env_missing_desc')) }) : null,
    !busy && env && st === 'ready' && env.in_use_is_env === false
      ? jsx('p', { className: 'text-[11px] leading-relaxed text-amber-500',
          children: t('yache.stt_env_other', maskPath(env.in_use_python || '', t('yache.path_who'))) }) : null,
    !busy && env && Number(env.free_gb) >= 0 && Number(env.free_gb) < 1.5
      ? jsx('p', { className: 'text-[11px] leading-relaxed text-amber-500',
          children: t('yache.stt_env_disk', String(env.free_gb)) }) : null,
    !busy && env && st !== 'ready' ? jsx(Button, {
      className: 'w-full', size: 'xs', onClick: () => install(false),
      children: t('yache.stt_env_install') }) : null,
    !busy && env && st === 'ready' ? jsx(Button, {
      className: 'w-full', size: 'xs', variant: 'outline',
      onClick: () => install(true), children: t('yache.stt_env_rebuild') }) : null,
    // A folder on disk? Offer the way back out — the loop closes both
    // ways (ready / deps / broken all have something to remove). Two-step
    // inline confirm, amber on the arming click — no dialog, per owner's
    // standing rule; amber-400 is the owner's alert color.
    !busy && env && st !== 'missing' ? jsx(Button, {
      className: 'w-full', size: 'xs', variant: 'outline',
      style: unarm === 1 ? { borderColor: '#fbbf24', color: '#fbbf24' } : undefined,
      onClick: uninstall,
      children: t(unarm === 1 ? 'yache.stt_env_uninstall_confirm'
                              : 'yache.stt_env_uninstall') }) : null,
    failed ? jsx('p', { className: 'whitespace-pre-wrap text-[11px] text-rose-500',
      children: (job && job.src === 'uninstall'
        ? t('yache.stt_env_unfail') : t('yache.stt_env_fail'))
        + String((job && job.error) || '') }) : null,
    failed && tail ? jsx('pre', {
      className: 'max-h-24 overflow-auto whitespace-pre-wrap break-all rounded bg-muted/50 p-1.5 text-[10px] leading-snug text-muted-foreground',
      children: tail }) : null,
  ] });
}

function SttBody({ ctx, t }) {
  const [cur, setCur] = useState(null);
  const [probing, setProbing] = useState(true);
  const [model, setModel] = useState('');
  const [dir, setDir] = useState('');
  const [device, setDevice] = useState('auto');
  const [lang, setLang] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const load = useCallback(() => {
    setProbing(true);
    ctx.rest('/stt/state', { timeoutMs: 130000 })
      .then((d) => {
        if (!d) return;
        setCur(d);
        if (d.ok !== false && d.engine_ok) {
          const m = String(d.model || 'small');
          if ((d.sizes || []).some((s) => s.name === m)) {
            setModel(m); setDir('');
          } else { setModel('__dir__'); setDir(m); }
          setDevice(['auto', 'cpu', 'cuda'].includes(String(d.device))
            ? String(d.device) : 'auto');
          setLang(String(d.language || ''));
        }
      })
      .catch((err) => setCur({ ok: false, engine_ok: false,
        error: String((err && err.message) || err) }))
      .finally(() => { setProbing(false); });
  }, [ctx]);
  useEffect(() => { load(); }, [load]);
  const [dl, setDl] = useState(false);
  const [dlp, setDlp] = useState(null);  // live snapshot from /stt/download/status
  const [delOpen, setDelOpen] = useState(false);
  // The chosen size entry from the state scan: cached=false keeps the
  // save button DISABLED and makes the Download button the only
  // actionable path — download failure = network, save failure =
  // engine, and a beginner can tell which button they pressed.
  const selSize = ((cur && cur.sizes) || [])
    .find((s) => s.name === model) || null;
  const needsDl = !!selSize && !selSize.cached;
  const doDownload = useCallback(async () => {
    if (dl) return;
    setDl(true); setNote(''); setDlp({ pct: 0 });
    // The POST only LAUNCHES the background job and answers at once;
    // the poller below then owns the finish line, which is what lets
    // the panel show live progress instead of a frozen "please wait".
    try {
      const r = await ctx.rest('/stt/download', { method: 'POST',
        body: { model } });
      if (r && r.ok) {
        if (r.already_cached || r.state === 'done') {
          setDl(false); setDlp(null);
          setNote(t('yache.stt_dl_cached', String(r.model || model)));
          load();
        }
      } else {
        setDl(false); setDlp(null);
        setNote(t('yache.stt_dl_fail')
          + String((r && (r.detail || r.error)) || 'HTTP error'));
      }
    } catch (err) {
      setDl(false); setDlp(null);
      setNote(t('yache.stt_dl_fail') + String((err && err.message) || err));
    }
  }, [ctx, dl, model, t, load]);
  // While a download is in flight, poll once a second for bytes-on-disk
  // progress: percent, MB/s and an ETA from the real measured speed.
  // done -> unlock Verify & save; error -> raw engine message; idle
  // (gateway restarted mid-download) -> stop the bar, the partial
  // cache simply resumes on the next Download click.
  useEffect(() => {
    if (!dl) return undefined;
    let stop = false;
    const tick = async () => {
      if (stop) return;
      try {
        const r = await ctx.rest('/stt/download/status');
        if (stop) return;
        if (r && r.state === 'done') {
          setDl(false); setDlp(null);
          setNote(t('yache.stt_dl_done', String(r.model || model)));
          load();
        } else if (r && r.state === 'error') {
          setDl(false); setDlp(null);
          setNote(t('yache.stt_dl_fail') + String(r.error || ''));
        } else if (r && r.state === 'running') {
          setDlp({ pct: Number(r.pct || 0), mbps: Number(r.mbps || 0),
            eta_s: (r.eta_s === null || r.eta_s === undefined)
              ? null : Number(r.eta_s) });
        } else if (r && r.state === 'idle') {
          setDl(false); setDlp(null);
        }
      } catch { /* transient transport blip: keep polling */ }
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => { stop = true; clearInterval(timer); };
  }, [dl, ctx, model, t, load]);
  // Reopening the panel mid-download resumes the live bar (the job is
  // owned by the gateway, not by this component).
  useEffect(() => {
    ctx.rest('/stt/download/status')
      .then((r) => { if (r && r.state === 'running') {
        setDl(true);
        setDlp({ pct: Number(r.pct || 0), mbps: Number(r.mbps || 0),
          eta_s: (r.eta_s === null || r.eta_s === undefined)
            ? null : Number(r.eta_s) });
      } })
      .catch(() => {});
  }, [ctx]);
  const pickDir = useCallback(async () => {
    try {
      const r = await ctx.os?.pickOpenPath?.({ title: t('yache.stt_dir_entry'),
        directories: true, defaultPath: dir || undefined });
      const p = typeof r === 'string' ? r
        : (r && (r.path || (Array.isArray(r.paths) ? r.paths[0] : null))) || null;
      if (p) { setDir(String(p)); setNote(''); }
    } catch { /* the typed path stays the fallback */ }
  }, [ctx, t, dir]);
  const save = useCallback(async () => {
    if (busy) return;
    const chosen = model === '__dir__' ? dir.trim() : model;
    if (!chosen) { setNote(t('yache.stt_need_model')); return; }
    setBusy(true); setNote('');
    try {
      const r = await ctx.rest('/stt/save', {
        method: 'POST',
        body: { model: chosen, language: lang, device },
        // An uncached size downloads inside this call (official source,
        // ~30 min for large-v3); stay generous — a half-cut download
        // would cost the user a full retry.
        timeoutMs: 1800000,
      });
      if (r && r.ok) {
        setNote(t('yache.stt_saved', String(r.model || chosen),
          String(r.device_used || device), String(r.load_s ?? '?')));
        load();
      } else {
        setNote(t('yache.stt_rejected')
          + String((r && (r.detail || r.error)) || 'HTTP error'));
      }
    } catch (err) {
      setNote(t('yache.stt_rejected') + String((err && err.message) || err));
    } finally { setBusy(false); }
  }, [ctx, busy, model, dir, lang, device, t, load]);
  // Deletion runs INSIDE ConfirmDialog's onConfirm: throwing keeps the
  // official dialog open with the error shown inline, a success rides
  // the dialog's own done-then-close beat and refreshes the cache list.
  const doDelete = useCallback(async () => {
    const r = await ctx.rest('/stt/delete', { method: 'POST',
      body: { model } });
    if (!r || !r.ok) {
      throw new Error(String((r && (r.detail || r.error))
        || 'HTTP error'));
    }
    setNote(t('yache.stt_del_ok', model, String(r.freed_gb ?? '?')));
    load();
  }, [ctx, model, t, load]);

  if (probing && !cur) {
    return jsx('p', { className: 'text-xs text-muted-foreground',
                      children: t('yache.stt_loading') });
  }
  if (!cur || cur.ok === false || cur.engine_ok === false) {
    return jsxs('div', { className: 'space-y-2', children: [
      jsxs('div', { className: 'rounded-md border border-rose-500/40 bg-rose-500/10 p-2.5',
        children: [
          jsxs('p', { className: 'text-xs font-semibold text-rose-500', children: [
            t('yache.stt_engine_down'),
            jsx('span', { className: 'ml-2 font-normal text-muted-foreground',
              children: String((cur && (cur.error || cur.detail)) || '') }),
          ] }),
          jsx('p', { className: 'mt-1 text-[11px] text-muted-foreground',
                     children: t('yache.stt_engine_tip') }),
        ] }),
            jsx(EnvRow, { ctx, t, onReady: load }),
jsx(Button, { size: 'xs', variant: 'outline', onClick: load,
        disabled: probing,
        children: probing ? t('yache.stt_loading') : t('yache.stt_retry') }),
    ] });
  }
  const rejected = note.startsWith(t('yache.stt_rejected'))
    || note.startsWith(t('yache.stt_dl_fail'));
  const sizeName = (s) => `${s.name} \u00b7 ${s.gb} GB \u00b7 `
    + (s.cached ? t('yache.stt_cached') : t('yache.stt_dl'));
  return jsxs('div', { className: 'space-y-3', children: [
    jsx('p', { className: 'text-xs leading-relaxed text-muted-foreground',
               children: t('yache.stt_desc') }),
    jsx(EnvRow, { ctx, t, onReady: load }),
    jsxs('div', { className: 'space-y-1', children: [
      jsx('label', { className: 'block text-xs text-muted-foreground',
                     children: t('yache.stt_model') }),
      jsxs(Select, { value: model,
        onValueChange: (v) => { setModel(v); setNote(''); }, children: [
        jsxs(SelectTrigger, { className: 'w-full', children: [
          jsx(SelectValue, { placeholder: t('yache.stt_model') })] }),
        jsxs(SelectContent, { children: [
          ...(cur.sizes || []).map((s) => jsx(SelectItem, { value: s.name,
            children: sizeName(s) }, s.name)),
          jsx(SelectItem, { value: '__dir__',
            children: t('yache.stt_dir_entry') }, '__dir__'),
        ] }),
      ] }),
      model === '__dir__' ? jsxs('div', { className: 'mt-1 space-y-1', children: [
        jsx('label', { className: 'block text-xs text-muted-foreground',
                       children: t('yache.stt_dir_label') }),
        jsxs('div', { className: 'flex items-center gap-1.5', children: [
          jsx(Input, { value: dir, placeholder: 'D:\\models\\whisper-large-v3-ct2',
                       onChange: (e) => setDir(e.target.value) }),
          jsx(Button, { size: 'xs', variant: 'outline', onClick: pickDir,
                        children: t('yache.stt_dir_browse') }),
        ] }),
      ] }) : null,
    ] }),
    jsxs('div', { className: 'grid grid-cols-2 gap-2', children: [
      jsxs('div', { className: 'space-y-1', children: [
        jsx('label', { className: 'block text-xs text-muted-foreground',
                       children: t('yache.stt_device') }),
        jsxs(Select, { value: device,
          onValueChange: (v) => { setDevice(v); setNote(''); }, children: [
          jsxs(SelectTrigger, { className: 'w-full', children: [
            jsx(SelectValue, {})] }),
          jsxs(SelectContent, { children: [
            jsx(SelectItem, { value: 'auto',
              children: t('yache.stt_dev_auto') }, 'auto'),
            jsx(SelectItem, { value: 'cpu', children: 'CPU' }, 'cpu'),
            jsx(SelectItem, { value: 'cuda',
              children: 'GPU (CUDA)' }, 'cuda'),
          ] }),
        ] }),
      ] }),
      jsxs('div', { className: 'space-y-1', children: [
        jsx('label', { className: 'block text-xs text-muted-foreground',
                       children: t('yache.stt_lang') }),
        jsxs(Select, { value: lang || 'auto',
          onValueChange: (v) => { setLang(v === 'auto' ? '' : v); setNote(''); },
          children: [
          jsxs(SelectTrigger, { className: 'w-full', children: [
            jsx(SelectValue, {})] }),
          jsxs(SelectContent, { children: [
            // Radix SelectItem rejects value='' — "auto" is the sentinel.
            jsx(SelectItem, { value: 'auto',
              children: t('yache.stt_lang_auto') }, 'auto'),
            ...(cur.langs || []).filter(Boolean).map((c) =>
              jsx(SelectItem, { value: c, children: STT_LANGS[c] || c }, c)),
          ] }),
        ] }),
      ] }),
    ] }),
    jsx('p', { className: 'text-[11px] text-muted-foreground',
      children: t('yache.stt_now', String((cur && cur.model) || ''),
        String((cur && cur.device) || 'auto'),
        String((cur && cur.language) || 'auto')) }),
    needsDl ? jsxs('div', { className: 'space-y-2', children: [
      jsx('p', { className: 'text-[11px] leading-relaxed text-amber-500',
                 children: t('yache.stt_dl_hint', String(selSize.gb)) }),
      dl ? jsxs('div', { className: 'space-y-1', children: [
        jsx('div', { className: 'h-2 w-full overflow-hidden rounded-full bg-muted',
          children: jsx('div', {
            className: 'h-full rounded-full bg-emerald-500 transition-all duration-700',
            style: { width: `${Math.max(1.5, Number(dlp && dlp.pct) || 0)}%` } }) }),
        jsx('p', { className: 'text-[10px] tabular-nums text-muted-foreground',
          children: t('yache.stt_dl_prog',
            (Number(dlp && dlp.pct) || 0).toFixed(1),
            (Number(dlp && dlp.mbps) || 0).toFixed(1),
            dlp && dlp.eta_s != null && dlp.eta_s > 0
              ? (dlp.eta_s >= 60
                ? `${Math.floor(dlp.eta_s / 60)}m${String(Math.round(dlp.eta_s % 60)).padStart(2, '0')}s`
                : `${Math.round(dlp.eta_s)}s`)
              : t('yache.stt_dl_eta_calc')) }),
      ] }) : null,
      jsx(Button, { className: 'w-full', variant: 'outline',
        disabled: dl || busy, onClick: doDownload,
        children: dl ? t('yache.stt_dl_busy')
                     : t('yache.stt_dl_btn', String(selSize.gb)) }),
    ] }) : null,
    selSize && selSize.cached && !dl ? jsx(Button, {
      className: 'w-full', size: 'xs', variant: 'outline',
      disabled: busy || String((cur && cur.model) || '') === model,
      onClick: () => { setNote(''); setDelOpen(true); },
      children: String((cur && cur.model) || '') === model
        ? t('yache.stt_del_active', model)
        : t('yache.stt_del_btn', String(selSize.gb)) }) : null,
    jsx(Button, { className: 'w-full',
      disabled: busy || dl || needsDl, onClick: save,
      children: busy ? t('yache.stt_verifying') : t('yache.stt_save') }),
    needsDl && !busy && !dl ? jsx('p', {
      className: 'text-[11px] text-muted-foreground',
      children: t('yache.stt_save_locked') }) : null,
    note ? jsx('p', { className: cn('whitespace-pre-wrap text-[11px]',
      rejected ? 'text-rose-500' : 'text-emerald-500'),
      children: note }) : null,
    jsx(ConfirmDialog, {
      open: delOpen, setOpen: setDelOpen,
      onClose: () => setDelOpen(false),
      onConfirm: doDelete,
      title: t('yache.stt_del_title'),
      desc: t('yache.stt_del_desc', String(model),
        String(selSize ? selSize.gb : '?')),
      confirmLabel: t('yache.stt_del_confirm'),
      busyLabel: t('yache.stt_del_busy'),
      doneLabel: t('yache.stt_del_done'),
      destructive: true,
    }),
  ] });
}

/* Wake & conversation: session mode, stop phrases and the experimental
   playback gate all live here now (the locked IA). */
/* Wake phrase: bilingual, and verified before anything is written. Saving
   builds a throwaway real engine on the backend — nothing persists unless
   that engine accepts the phrase, and its own error text comes back
   verbatim. The backend auto-picks the local KWS model matching the phrase
   script (CJK -> pinyin model, latin -> spelled model), so one machine can
   wake in both languages. An armed listener keeps the old word until a wake
   off->on re-arm (same freeze policy as the session-mode switch below), and
   the save note spells that out. */
function WakePhrase({ ctx, t }) {
  const [cur, setCur] = useState(null);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');

  useEffect(() => {
    let live = true;
    ctx.rest('/wake-phrase', { timeoutMs: 8000 })
      .then((d) => {
        if (!live || !d || d.ok === false) return;
        setCur(d);
        setDraft((v) => (v === '' ? String(d.phrase || '') : v));
      })
      .catch(() => { /* empty first paint; the save path surfaces errors */ });
    return () => { live = false; };
  }, [ctx]);

  const save = useCallback(async () => {
    const phrase = draft.trim();
    if (!phrase || busy) return;
    setBusy(true);
    setNote('');
    try {
      const r = await ctx.rest('/wake-phrase',
        { method: 'POST', body: { phrase }, timeoutMs: 90000 });
      if (r && r.ok) {
        setCur(r);
        setDraft(String(r.phrase || phrase));
        setNote((r.apply === 'rearm-toggle' ? t('yache.wp_saved_rearm')
          : t('yache.wp_saved_auto'))
          + (r.model_changed ? t('yache.wp_model_switched') : '')
          + t('yache.wp_name_sync'));
        try {
          queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] });
        } catch { /* best-effort host config-cache refresh */ }
      } else {
        setNote(t('yache.wp_rejected') + ' ' + String((r && (r.detail || r.error)) || ''));
      }
    } catch (err) {
      setNote(t('yache.wp_rejected') + ' ' + String((err && err.message) || err));
    } finally {
      setBusy(false);
    }
  }, [ctx, draft, busy, t]);

  const dirty = !!draft.trim() && (!cur || draft.trim() !== String(cur.phrase || ''));
  const fail = !!note && note.startsWith(t('yache.wp_rejected'));
  const kind = !cur ? t('yache.wp_kind_unknown')
    : cur.model_kind === 'zh' ? t('yache.wp_kind_zh')
      : cur.model_kind === 'en' ? t('yache.wp_kind_en') : t('yache.wp_kind_unknown');
  return jsxs('div', { children: [
    jsx('div', { className: 'text-xs font-semibold', children: t('yache.wp_title') }),
    jsx('p', { className: 'mb-2 mt-1 text-[11px] leading-relaxed text-muted-foreground',
      children: t('yache.wp_desc') }),
    jsxs('p', { className: 'mb-2 text-[10px] text-muted-foreground', children: [
      t('yache.wp_now', kind, cur && cur.listening ? t('yache.wp_armed') : t('yache.wp_idle')),
      cur && cur.language === 'mixed' ? t('yache.wp_mixed_note') : null,
    ] }),
    jsxs('div', { className: 'flex items-center gap-2', children: [
      jsx(Input, { value: draft, disabled: busy, placeholder: t('yache.wp_label'),
        onChange: (e) => setDraft(e.target.value), className: 'h-8 flex-1 text-xs' }),
      jsx(Button, { size: 'xs', disabled: busy || !dirty, onClick: () => void save(),
        children: busy ? t('yache.wp_verifying') : t('yache.wp_save') }),
    ] }),
    note ? jsx('p', { className: 'mt-1 whitespace-pre-wrap text-[11px] '
      + (fail ? 'text-rose-500' : 'text-muted-foreground'), children: note }) : null,
  ] });
}

function WakeBody({ ctx, t, st }) {
  return jsxs('div', { className: 'space-y-3', children: [
    jsx(WakePhrase, { ctx, t }),
    jsx('div', { className: 'border-t border-border' }),
    jsx(WakeSession, { ctx, t }),
    jsx('div', { className: 'border-t border-border' }),
    jsx(StopPhrase, { ctx, t }),
    st ? jsxs('div', { className: 'border-t border-border pt-1', children: [
      StatusRow({ label: t('yache.gate'), value: st.wake_gate ? 'on' : 'off',
                  title: t('yache.gate_tip') }),
    ] }) : null,
  ] });
}

/* Cloning section — the voice library itself: built-in speakers, cloned
   voices and designed voices in one picker; preview before committing;
   saving runs a real synthesis first and only then writes. The saved
   selection is exactly what unlocks the TTS synthesis section, because
   tier/quality can only be judged against a voice that actually exists. */
/* The wizard ledger has three transient states — bytes flowing (`running`),
   source speed test (`probing`) and CUDA pack unpack + DLL deploy
   (`extracting`) — and the progress poll must stay alive through ALL of
   them, not just `running`. When it died on the others, `done` was never
   seen and the accelerator light only lit after closing and reopening the
   panel (owner report, 11-19). Anything else in the ledger is final. */
const DL_ACTIVE = (s) => s === 'running' || s === 'probing' || s === 'extracting';
function DownloadSection({ ctx, t, lib, refreshLib }) {
  // Guided download for users with nothing local yet: two dropdowns
  // (model x quantization) with a live size/memory estimate, a folder
  // picker, an auto-probed download source, one progress bar, resume and
  // cancel — plus the optional CUDA accelerator pack, whose button stays
  // locked until a model pair actually scans clean (backend-computed).
  const [combos, setCombos] = useState(null);
  const [wv, setWv] = useState('');
  const [wq, setWq] = useState('');
  const [wdir, setWdir] = useState('');
  const [srcs, setSrcs] = useState(null);   // {phase:'probe'|'done'|'fail', label}
  const [dlp, setDlp] = useState(null);     // wizard/status payload
  const [gpu, setGpu] = useState(null);     // wizard/gpu payload
  const [canceling, setCanceling] = useState(false);
  const [uninst, setUninst] = useState(0);    // 0 idle · 1 armed (second click confirms)
  const [note, setNote] = useState('');
  const running = !!dlp && dlp.state === 'running';
  const eta = (s) => s == null || s <= 0 ? t('yache.wiz_eta_calc')
    : (s >= 60 ? `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, '0')}s`
               : `${Math.round(s)}s`);
  const loadGpu = useCallback(() => {
    ctx.rest('/wizard/gpu').then((r) => { if (r) setGpu(r); }).catch((e) => dbg('wizard/gpu load', e));
  }, []);
  useEffect(() => {
    // catalog is static; fire once. default the dropdowns to 1.7B/customvoice
    // + q4 (the recommended beginner pair) when it lands.
    ctx.rest('/wizard/catalog').then((r) => {
      const list = (r && r.combos) || [];
      if (!list.length) return;
      setCombos(list);
      setWv((v) => v || (list.find((c) => c.combo === '1.7b-customvoice') || list[0]).combo);
      setWq((v) => v || (list.find((c) => c.combo === (wv || '1.7b-customvoice') && c.quant === 'q4_k_m') || list[0]).quant);
    }).catch((e) => dbg('wizard/catalog', e));
    setSrcs({ phase: 'probe' });
    ctx.rest('/wizard/source').then((r) => {
      if (r && r.endpoint) setSrcs({ phase: 'done', label: String(r.endpoint).replace(/^https?:\/\//, '') });
      else setSrcs({ phase: 'fail', label: String((r && r.error) || '') });
    }).catch((e) => { dbg('wizard/source probe', e); setSrcs({ phase: 'fail', label: '' }); });
    loadGpu();
    ctx.rest('/wizard/status').then((r) => { if (r && r.state && r.state !== 'idle') setDlp(r); }).catch((e) => dbg('wizard/status boot', e));
  }, []);
  // While any download is in a transient state, poll the one-second
  // progress — including `extracting`: the CUDA pack's last beat is a
  // ~15 s DLL deploy, and the old `running`-only condition died exactly
  // there, so `done` (and with it loadGpu/refreshLib) never fired in
  // place. Terminal states stop the poll; `done` refreshes the lights.
  useEffect(() => {
    if (!dlp || !DL_ACTIVE(dlp.state)) return;
    const iv = setInterval(() => {
      ctx.rest('/wizard/status').then((r) => {
        if (!r) return;
        setDlp(r);
        if (r.state === 'done') {
          setNote(String(r.target || '').includes('yache-cuda')
            ? t('yache.wiz_gpu_done') : t('yache.wiz_done'));
          refreshLib(); loadGpu();
        } else if (r.state === 'cancelled' || r.state === 'interrupted') {
          loadGpu();
        }
      }).catch(() => {});
    }, 1000);
    return () => clearInterval(iv);
  }, [!!dlp && DL_ACTIVE(dlp.state)]);
  const startProbe = () => {
    setSrcs({ phase: 'probe' });
    ctx.rest('/wizard/source').then((r) => {
      if (r && r.endpoint) setSrcs({ phase: 'done', label: String(r.endpoint).replace(/^https?:\/\//, '') });
      else setSrcs({ phase: 'fail', label: String((r && r.error) || '') });
    }).catch((e) => { dbg('wizard/source probe', e); setSrcs({ phase: 'fail', label: '' }); });
  };
  const list = combos || [];
  const models = [];
  list.forEach((c) => { if (!models.some((m) => m.label === c.model_label)) models.push({ value: c.combo, label: c.model_label }); });
  const combo = list.find((c) => c.combo === wv);
  const quants = list.filter((c) => c.combo === wv);
  const qSel = quants.find((c) => c.quant === wq) || quants[0];
  const start = async () => {
    setNote('');
    try {
      const r = await ctx.rest('/wizard/model', { method: 'POST',
        body: { combo: wv, quant: (qSel && qSel.quant) || 'q4_k_m', target: wdir.trim() },
        timeoutMs: 30000 });
      if (r && r.ok) setDlp({ state: 'running' });
      else setNote(String((r && r.error) || 'failed'));
    } catch (e) { dbg('wizard action', e); setNote(String((e && e.message) || e)); }
  };
  const resume = async () => {
    setNote('');
    try {
      const r = await ctx.rest('/wizard/resume', { method: 'POST', body: {}, timeoutMs: 30000 });
      if (r && r.ok) setDlp({ state: 'running' });
      else setNote(String((r && r.error) || 'failed'));
    } catch (e) { dbg('wizard action', e); setNote(String((e && e.message) || e)); }
  };
  const cancel = async () => {
    setCanceling(true);
    try { await ctx.rest('/wizard/cancel', { method: 'POST', body: {}, timeoutMs: 15000 }); } catch (e) { dbg('wizard/cancel', e); }
  };
  const startGpu = async () => {
    setNote('');
    try {
      const r = await ctx.rest('/wizard/gpu', { method: 'POST', body: {}, timeoutMs: 30000 });
      if (r && r.ok) setDlp({ state: 'running' });
      else setNote(String((r && r.error) || 'failed'));
    } catch (e) { dbg('wizard action', e); setNote(String((e && e.message) || e)); }
  };
  const uninstallGpu = async () => {
    // Two-step INLINE confirm (the owner dislikes dialogs): the first click
    // arms the button, the second within 3 s really takes the pack off.
    // The backend reverse-uninstalls along the LEDGER's recorded install
    // path — the DLL root on a user's machine is wherever the installer
    // wrote it back then, not what we could guess today.
    if (uninst === 0) {
      setUninst(1);
      setTimeout(() => setUninst((u) => (u === 1 ? 0 : u)), 3000);
      return;
    }
    setUninst(0); setNote('');
    try {
      const r = await ctx.rest('/wizard/gpu', { method: 'DELETE', timeoutMs: 30000 });
      if (r && r.ok) { setNote(t('yache.wiz_uninstalled')); refreshLib(); loadGpu(); }
      else {
        const WHY = { busy: t('yache.wiz_uninstall_busy'),
                      in_use: t('yache.wiz_uninstall_in_use'),
                      not_pack: t('yache.wiz_uninstall_not_pack'),
                      io: t('yache.wiz_uninstall_io') };
        setNote(WHY[r && r.reason] || String((r && r.error) || 'failed'));
      }
    } catch (e) { dbg('wizard gpu uninstall', e); setNote(String((e && e.message) || e)); }
  };
  useEffect(() => {
    // cancel button feedback: poll until the ledger really moved, then
    // restore the button greyed for one beat (owner's 11-10 convention).
    if (!canceling) return;
    let tries = 0;
    const iv = setInterval(async () => {
      tries += 1;
      let r = null;
      try { r = await ctx.rest('/wizard/status'); } catch (e) { dbg('wizard/cancel poll', e); }
      if ((r && !DL_ACTIVE(r.state)) || tries > 30) {
        clearInterval(iv); setCanceling(false);
        if (r) setDlp(r);
        loadGpu();
      }
    }, 1000);
    return () => clearInterval(iv);
  }, [canceling]);
  const shown = dlp && (dlp.state === 'running' || dlp.state === 'extracting'
    || dlp.state === 'probing' || dlp.state === 'error'
    || dlp.state === 'cancelled' || dlp.state === 'interrupted'
    || dlp.state === 'installed_unverified') ? dlp : null;
  const canStart = !!wdir.trim() && !!qSel && !running;
  return jsxs('div', { className: 'space-y-1.5 rounded-lg border border-border p-2', children: [
    jsx('p', { className: 'text-[11px] font-medium leading-relaxed text-amber-400',
      children: t('yache.wiz_top_tip') }),
    jsx('div', { className: 'text-[11px] font-medium text-muted-foreground',
      children: t('yache.wiz_title') }),
    jsxs('div', { className: 'grid grid-cols-2 gap-1.5', children: [
      jsxs('div', { className: 'space-y-1', children: [
        jsx('label', { className: 'block text-xs text-muted-foreground',
                       children: t('yache.wiz_model') }),
        jsxs(Select, { value: wv, disabled: running, onValueChange: (v) => {
            setWv(v);
            const qs = list.filter((c) => c.combo === v);
            const dq = qs.find((c) => c.quant === 'q4_k_m') || qs[0];
            if (dq) setWq(dq.quant);
          }, children: [
          jsx(SelectTrigger, { className: 'w-full', children: jsx(SelectValue, {}) }),
          jsxs(SelectContent, { children: models.map((m) =>
            jsx(SelectItem, { value: m.value, children: m.label }, m.value)) }),
        ] }),
      ] }),
      jsxs('div', { className: 'space-y-1', children: [
        jsx('label', { className: 'block text-xs text-muted-foreground',
                       children: t('yache.wiz_quant_l') }),
        jsxs(Select, { value: (qSel && qSel.quant) || '', disabled: running,
          onValueChange: setWq, children: [
          jsx(SelectTrigger, { className: 'w-full', children: jsx(SelectValue, {}) }),
          jsxs(SelectContent, { children: quants.map((c) =>
            jsx(SelectItem, { value: c.quant, children: c.label }, c.quant)) }),
        ] }),
      ] }),
    ] }),
    qSel ? jsx('p', { className: 'text-[11px] leading-relaxed text-amber-400',
      children: t('yache.wiz_size', `${qSel.total_gb}`, `${qSel.vram_gb}`) }) : null,
    jsxs('div', { className: 'flex items-center gap-1.5', children: [
      jsx('p', { className: 'flex-1 text-[10px] leading-4 text-muted-foreground',
        children: !srcs ? '' : srcs.phase === 'probe' ? t('yache.wiz_src_probing')
          : srcs.phase === 'fail' ? t('yache.wiz_src_fail')
          : t('yache.wiz_src', srcs.label) }),
      jsx(Button, { size: 'xs', variant: 'outline', disabled: running,
        onClick: startProbe, children: t('yache.wiz_probe') }),
    ] }),
    jsxs('div', { className: 'flex items-center gap-1.5', children: [
      jsx(Input, { value: wdir, className: 'h-8 flex-1 text-xs font-mono',
        placeholder: t('yache.wiz_dir_ph'), disabled: running,
        onChange: (e) => setWdir(e.target.value) }),
      jsx(Button, { size: 'xs', variant: 'outline', disabled: running,
        onClick: async () => {
          try {
            const r = await ctx.os.pickOpenPath({ title: t('yache.wiz_dir'),
              directories: true, defaultPath: wdir || (lib && lib.active_dir && lib.active_dir.path) || '' });
            const p = typeof r === 'string' ? r
              : (r && (r.path || r.filePath || r.filePaths && r.filePaths[0])) || '';
            if (p) setWdir(String(p));
          } catch (e) { dbg('wizard/browse', e); }
        }, children: t('yache.wiz_browse') }),
    ] }),
    shown && DL_ACTIVE(shown.state) ? jsxs('div', { className: 'space-y-1', children: [
      jsx('div', { className: 'h-2 w-full overflow-hidden rounded-full bg-muted',
        children: jsx('div', {
          className: 'h-full rounded-full bg-emerald-500 transition-all duration-700',
          style: { width: `${Math.max(1.5, Number(shown.pct) || 0)}%` } }) }),
      shown.state !== 'extracting' ? jsx('p', { className: 'text-[10px] tabular-nums text-muted-foreground',
        children: t('yache.wiz_prog', (Number(shown.pct) || 0).toFixed(1),
          (Number(shown.mbps) || 0).toFixed(1), eta(shown.eta_s)) }) : null,
      jsx('p', { className: 'text-[10px] text-muted-foreground',
        children: String(shown.step_label || '') }),
      // DLL deploy is quiet work with nothing to cancel — the backend
      // ignores a cancel at this beat, so don't offer a button that lies.
      shown.state === 'extracting' ? jsx('p', { className: 'text-[11px] font-medium text-amber-400',
        children: t('yache.wiz_extracting') }) : null,
      shown.state === 'running' ? jsx(Button, { className: 'w-full', size: 'xs', variant: 'outline',
        disabled: canceling, onClick: cancel,
        children: canceling ? t('yache.wiz_cancelling') : t('yache.wiz_cancel') }) : null,
    ] }) : null,
    shown && !DL_ACTIVE(shown.state) ? jsxs('div', { className: 'space-y-1', children: [
      jsx('p', { className: `text-[11px] ${shown.state === 'error' ? 'text-rose-500' : 'text-muted-foreground'}`,
        children: `${shown.state === 'error' ? String(shown.error || '') : String(shown.state)}` }),
      shown.state === 'error' || shown.state === 'interrupted' || shown.state === 'cancelled'
        ? jsx(Button, { className: 'w-full', variant: 'outline',
            disabled: running, onClick: resume, children: t('yache.wiz_resume') }) : null,
    ] }) : null,
    !running ? jsx(Button, { className: 'w-full', disabled: !canStart,
      onClick: start, children: t('yache.wiz_start') }) : null,
    note ? jsx('p', { className: 'text-[11px] text-rose-500', children: note }) : null,
    // ---- GPU accelerator pack ------------------------------------------------
    jsxs('div', { className: 'space-y-1 border-t border-border pt-1.5', children: [
      jsx('div', { className: 'text-[11px] font-medium text-muted-foreground',
        children: t('yache.wiz_gpu_title') }),
      jsx('p', { className: `text-[11px] leading-4 ${gpu && gpu.installed ? 'text-emerald-500' : 'text-muted-foreground'}`,
        children: !gpu ? '' : gpu.installed ? t('yache.wiz_gpu_on') : t('yache.wiz_gpu_off') }),
      // Installed? offer the way back out — the loop closes both ways.
      // Armed state reads amber (owner's alert color), not a dialog.
      gpu && gpu.installed
        ? jsx(Button, { className: 'w-full', size: 'xs', variant: 'outline',
            disabled: running,
            style: uninst === 1
              ? { pointerEvents: 'auto', borderColor: '#fbbf24', color: '#fbbf24' }
              : { pointerEvents: 'auto' },
            onClick: uninstallGpu,
            children: t(uninst === 1 ? 'yache.wiz_uninstall_confirm' : 'yache.wiz_uninstall') })
        : null,
      !gpu || gpu.installed ? null
        : jsx(Button, { className: 'w-full', size: 'xs',
            variant: gpu.gate_ok ? 'default' : 'outline',
            style: gpu.gate_ok
              ? { background: '#fbbf24', borderColor: '#fbbf24', color: '#000' }
              : { pointerEvents: 'auto' },
            disabled: !gpu.gate_ok || running,
            title: gpu.gate_ok ? '' : gpu.gate_reason ? String(gpu.gate_reason) : '',
            onClick: startGpu,
            children: t('yache.wiz_gpu_btn', String(gpu.size_gb || '0.7')) }),
      gpu && !gpu.installed
        ? jsx('p', { className: 'text-[10px] leading-4 text-amber-400',
            children: t('yache.wiz_gpu_nvidia') }) : null,
      gpu && !gpu.installed && !gpu.gate_ok
        ? jsx('p', { className: 'text-[10px] text-muted-foreground',
            children: t('yache.wiz_gpu_locked') }) : null,
      jsx('p', { className: 'text-[10px] leading-4 text-muted-foreground',
        children: t('yache.wiz_gpu_note') }),
    ] }),
  ] });
}

function CloneBody({ ctx, t, lib, refreshLib }) {
  // Model granularity is the third axis of a build (asked 2026-11-18):
  // one "Base" label used to hide 0.6b and 1.7b files wearing the same
  // quant names at very different VRAM costs. The variant menu now lists one
  // entry per (variant, size) the scan really found; files without a size
  // token read as the big build, so pre-split directories keep behaving.
  const combos = [];
  const rawScan = lib.scan || {};
  ['customvoice', 'base', 'voicedesign'].forEach((v) => {
    const e = rawScan[v];
    if (!e || !e.present) return;
    const sizes = [];
    (e.quants || []).forEach((x) => {
      const z = String((x && typeof x === 'object' ? x.model_size : '') || '')
        .trim().toLowerCase() === '0.6b' ? '0.6b' : '1.7b';
      if (sizes.indexOf(z) < 0) sizes.push(z);
    });
    if (!sizes.length) sizes.push('1.7b');
    sizes.sort((a, b) => (parseFloat(b) || 0) - (parseFloat(a) || 0));
    sizes.forEach((z) => combos.push({ v, size: z }));
  });
  const present = combos.map((o) => o.v).filter((v, i, a) => a.indexOf(v) === i);
  const ckey = (v, z) => `${v}|${z}`;
  const [vt, setVt] = useState(null);          // combo key "variant|size"
  const combo = combos.find((o) => ckey(o.v, o.size) === vt) || null;
  const curVt = combo ? combo.v : null;
  const curSize = combo ? combo.size : '';
  const [sel, setSel] = useState('');
  const [instruct, setInstruct] = useState('');
  const [text, setText] = useState('');
  const [busy, setBusy] = useState('');   // 'prev' | 'save' | 'clone'
  const [note, setNote] = useState('');
  const [showNew, setShowNew] = useState(false);
  const [nc, setNc] = useState({ name: '', ref_audio: '', ref_text: '', description: '' });
  const [asr, setAsr] = useState('');  // '' | busy | ok | fail — reference transcript state
  // A design voice is re-invented on every call — only the preview just
  // heard is concrete. It rides here (b64 + the words it spoke + the spec
  // it ran on) so Export/Adopt act on THAT voice, even after the inputs
  // move on. adoptName is the user's chosen name for clone enrollment.
  const [clip, setClip] = useState(null);
  const [adoptName, setAdoptName] = useState('');
  const [dzBusy, setDzBusy] = useState('');   // '' | 'export' | 'adopt'
  // Deregistration guard: the name armed for the second (confirming) click;
  // it re-arms to '' after 3 s so a stray click can never silently delete.
  const [delArm, setDelArm] = useState('');
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current) return;
    const sv = lib.saved_voice;
    if (sv && (sv.kind || sv.variant)) {
      seeded.current = true;
      const sz = String(sv.model_size || '').trim().toLowerCase() === '0.6b'
        ? '0.6b' : '1.7b';
      if (sv.kind === 'speaker' && sv.speaker) {
        setVt(ckey('customvoice', sz)); setSel('spk:' + sv.speaker);
      } else if (sv.kind === 'clone' && sv.voice) {
        setVt(ckey('base', sz)); setSel('cln:' + sv.voice);
      } else if (sv.kind === 'design') {
        setVt(ckey('voicedesign', sz)); setSel('design');
        if (sv.instruct) setInstruct(String(sv.instruct));
      }
    }
  }, [lib]);
  const vtList = combos.map((o) => ({
    v: ckey(o.v, o.size),
    label: `${t(o.v === 'customvoice' ? 'yache.vt_cv'
      : o.v === 'base' ? 'yache.vt_base' : 'yache.vt_vd')
      .split('（')[0].split('(')[0].trimEnd()} · ${o.size.toUpperCase()}` }));
  const options = [];
  if (curVt === 'customvoice') {
    (lib.speakers || []).forEach((s) =>
      options.push({ v: 'spk:' + s, label: t('yache.grp_spk', s) }));
  } else if (curVt === 'base') {
    // Existing reference voices (clones) — pick one and it just works.
    (lib.clones || []).forEach((c) => {
      if (c.ref_ok) options.push({ v: 'cln:' + c.name, label: t('yache.grp_cln', c.name) });
    });
  }
  const buildSpec = () => {
    // model_size rides with every draft spec: preview and save must speak
    // with the exact build the combo names, not whatever the scanner lists
    // first under that variant label
    if (sel === 'design') {
      return { kind: 'design', variant: 'voicedesign', model_size: curSize,
               instruct: instruct.trim() };
    }
    if (sel.indexOf('spk:') === 0) {
      return { kind: 'speaker', variant: 'customvoice', model_size: curSize,
               speaker: sel.slice(4) };
    }
    return { kind: 'clone', variant: 'base', model_size: curSize,
             voice: sel.slice(4) };
  };
  const designMissing = sel === 'design' && !instruct.trim();
  const run = useCallback(async (path, body, mode, doneMsg, onOk) => {
    setBusy(mode); setNote('');
    try {
      const r = await ctx.rest(path, { method: 'POST', body,
        timeoutMs: mode === 'prev' ? 180000 : 300000 });
      if (r && r.ok) {
        if (doneMsg) setNote(doneMsg);
        else if (r.seconds != null && r.elapsed != null) {
          setNote(t('yache.aud_ok', r.seconds, r.elapsed));
        }
        playWav(r.wav_b64);
        if (onOk) onOk(r);
        if (refreshLib) refreshLib();
      } else {
        setNote(t('yache.voice_fail') + String((r && r.error) || ''));
      }
    } catch (err) {
      setNote(t('yache.voice_fail') + String((err && err.message) || err));
    } finally { setBusy(false); }
  }, [ctx, t, refreshLib]);
  const preview = () => run('/voice/preview',
    { spec: buildSpec(), text: text.trim() }, 'prev', null,
    sel === 'design'
      ? (r) => setClip({ wav: r.wav_b64, text: text.trim(),
                         spec: buildSpec() })
      : undefined);
  const save = () => run('/voice/save',
    { spec: buildSpec(), text: text.trim() }, 'save', t('yache.voice_saved'));
  const createClone = () => {
    const name = nc.name.trim();
    if (!name || !nc.ref_audio.trim()) return;
    setShowNew(false);
    run('/voice/clone', { name, ref_audio: nc.ref_audio.trim(),
      ref_text: nc.ref_text, description: nc.description,
      model_size: curSize }, 'clone', t('yache.voice_saved'));
    setVt(ckey('base', curSize || '1.7b')); setSel('cln:' + name); setAsr('');
  };
  // Export keeps the random voice you just heard: the design path re-rolls
  // a new voice on every call, so the preview clip is the only instance of
  // it. A native save dialog lets the user name and place the wav (older
  // hosts fall back to the plugin's own voices/design folder, chosen by the
  // backend). Adopt then enrolls that same clip through the clone path
  // under a name the user typed — which turns the designed voice into a
  // permanent one. Both act on the stored clip, never on the live fields.
  const dzExport = (body) => ctx.rest('/voice/design/export', {
    method: 'POST', body, timeoutMs: 60000 });
  const exportClip = async () => {
    if (!clip) { setNote(t('yache.dz_need_clip')); return; }
    setDzBusy('export'); setNote('');
    try {
      let dest = '';
      if (ctx.os && ctx.os.pickSavePath) {
        const picked = await ctx.os.pickSavePath({
          title: t('yache.dz_export'),
          defaultPath: 'designed-voice.wav',
          filters: [{ name: 'wav', extensions: ['wav'] }] });
        if (!picked) { setDzBusy(''); return; }   // dialog dismissed
        dest = picked;
      }
      const r = await dzExport({ wav_b64: clip.wav, path: dest,
                                 ref_text: clip.text });
      if (!r || !r.ok) {
        throw new Error((r && r.error) || t('yache.voice_fail')); }
      setNote('\u2705 ' + t('yache.dz_saved') + (r.path || dest));
      // default-folder saves are hard to find — reveal the file itself
      if (!dest && ctx.os && ctx.os.revealPath) {
        try { await ctx.os.revealPath(r.path); } catch { /* older hosts */ }
      }
    } catch (e) {
      setNote(t('yache.voice_fail') + String((e && e.message) || e));
    } finally { setDzBusy(''); }
  };
  const adoptClip = async () => {
    if (!clip) { setNote(t('yache.dz_need_clip')); return; }
    const name = adoptName.trim();
    if (!name) { setNote(t('yache.dz_need_name')); return; }
    setDzBusy('adopt'); setNote('');
    try {
      const w = await dzExport({ wav_b64: clip.wav, path: '',
                                 ref_text: clip.text });
      if (!w || !w.ok) {
        throw new Error((w && w.error) || t('yache.voice_fail')); }
      const r = await ctx.rest('/voice/clone', { method: 'POST',
        body: { name, ref_audio: w.path, ref_text: clip.text,
                description: String((clip.spec && clip.spec.instruct) || '')
                  .slice(0, 140),
                model_size: (clip.spec && clip.spec.model_size) || '' },
        timeoutMs: 300000 });
      if (!r || r.ok === false) {
        throw new Error((r && r.error) || t('yache.voice_fail')); }
      setAdoptName('');
      setNote('\u2705 ' + t('yache.dz_adopted') + name);
      if (refreshLib) refreshLib();
      setVt(ckey('base', (clip.spec && clip.spec.model_size) || curSize || '1.7b'));
      setSel('cln:' + name); setAsr('');
    } catch (e2) {
      setNote(t('yache.voice_fail') + String((e2 && e2.message) || e2));
    } finally { setDzBusy(''); }
  };
  // Backing out of a registry entry: the backend drops it, clears the
  // current-voice pointer if it pointed here, and deliberately keeps the
  // wav on disk (deleting recordings is the user's own call). Errors from
  // the route come back as thrown details, so no-throw means done.
  const dropClone = async (name) => {
    setBusy('drop'); setNote('');
    try {
      await ctx.rest('/voice/clone/' + encodeURIComponent(name),
        { method: 'DELETE', timeoutMs: 30000 });
      setNote('\u2705 ' + t('yache.dz_dropped') + name +
        t('yache.dz_dropped_wav'));
      setSel(''); setAsr('');
      if (refreshLib) refreshLib();
    } catch (e) {
      setNote(t('yache.voice_fail') + String((e && e.message) || e));
    } finally { setDelArm(''); setBusy(''); }
  };
  const pickRef = useCallback(async () => {
    setAsr('');
    try {
      const r = await ctx.os?.pickOpenPath?.({
        title: t('yache.clone_ref_pick'),
        filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'm4a', 'ogg'] }],
      });
      const p = typeof r === 'string' ? r
        : (r && (r.path || (Array.isArray(r.paths) ? r.paths[0] : null))) || null;
      if (!p) return;
      const path = String(p);
      const stem = path.replace(/[\\/]+$/, '').split(/[\\/]/).pop()
        ?.replace(/\.[^.]+$/, '') || '';
      setNc((v) => ({ ...v, ref_audio: path, name: v.name || stem }));
      setAsr('busy');
      try {
        const tr = await ctx.rest('/voice/ref/transcribe', {
          method: 'POST', body: { path }, timeoutMs: 120000 });
        if (tr && tr.ok && String(tr.text || '').trim()) {
          setNc((v) => ({ ...v, ref_text: String(tr.text).trim() }));
          setAsr('ok');
        } else setAsr('fail');
      } catch { setAsr('fail'); }
    } catch { /* the typed path stays the fallback */ }
  }, [ctx, t]);
  const dirPath = (lib && lib.active_dir && lib.active_dir.path) || '';
  const [dir, setDir] = useState(dirPath);
  // A path that was typed but never scanned must NOT inherit the previous
  // directory's scan results — the variant menus stay locked until
  // "read" confirms what this folder really holds.
  const savedPath = ((lib && lib.active_dir && lib.active_dir.path) || '').trim();
  const dirDirty = !!savedPath && dir.trim() !== savedPath;
  useEffect(() => { if (dirPath) setDir(dirPath); }, [dirPath]);
  const applyDir = useCallback(async (mode, pathOverride) => {
    setBusy('dir'); setNote('');
    try {
      const r = await ctx.rest('/voice/models/scan', {
        method: 'POST',
        body: mode === 'reset' ? { reset: true }
          : { path: String(pathOverride != null ? pathOverride : dir).trim() },
        timeoutMs: 60000,
      });
      if (r && r.ok) {
        setNote(mode === 'reset' ? t('yache.mdl_reset_ok')
          : t('yache.mdl_ok', (r.found_variants || []).length));
        if (refreshLib) refreshLib();
      } else {
        setNote(t('yache.voice_fail') + String((r && r.error) || ''));
      }
    } catch (err) {
      setNote(t('yache.voice_fail') + String((err && err.message) || err));
    } finally { setBusy(''); }
  }, [ctx, t, dir, refreshLib]);
  const pickDir = useCallback(async () => {
    try {
      const r = await ctx.os?.pickOpenPath?.({ title: t('yache.mdl_pick'),
        directories: true, defaultPath: dir || undefined });
      const p = typeof r === 'string' ? r
        : (r && (r.path || (Array.isArray(r.paths) ? r.paths[0] : null))) || null;
      if (p) { setDir(String(p)); applyDir('save', String(p)); }
    } catch { /* the typed path stays the fallback */ }
  }, [ctx, t, applyDir]);
  const missing = lib ? Object.entries(lib.scan || {})
    .filter(([, v]) => !(v && v.present)).map(([k]) => k) : [];
  const selChanged = (v) => {
    setSel(v); setNote('');
    if (v === 'design') {
      if (lib && lib.saved_voice && lib.saved_voice.kind === 'design') {
        setInstruct(String(lib.saved_voice.instruct || ''));
      }
      return;
    }
    // Picking an existing voice applies IT immediately (the backend
    // saves the pointer and mirrors the registry in the same call).
    // The confirmation line under the box states exactly that, so
    // selection = activation with zero ambiguity for any user.
    const spec = v.indexOf('spk:') === 0
      ? { kind: 'speaker', variant: 'customvoice', speaker: v.slice(4) }
      : { kind: 'clone', variant: 'base', voice: v.slice(4) };
    const msg = v.indexOf('spk:') === 0
      ? t('yache.pick_applied_spk', v.slice(4))
      : t('yache.pick_applied_clone', v.slice(4));
    void run('/voice/save', { spec, text: '' }, 'save', msg);
  };
  return jsxs('div', { className: 'space-y-2', children: [
    jsx(DownloadSection, { ctx, t, lib, refreshLib }),
    jsxs('div', { className: 'space-y-1.5 rounded-lg border border-border p-2', children: [
      jsx('label', { className: 'text-[11px] font-medium text-muted-foreground',
        children: t('yache.mdl_title') }),
      jsx('p', { className: 'text-[11px] leading-4 text-muted-foreground',
        children: t('yache.mdl_hint') }),
      jsxs('div', { className: 'flex items-center gap-1.5', children: [
        jsx(Input, { value: dir, className: 'h-8 flex-1 text-xs font-mono',
          placeholder: t('yache.mdl_ph'),
          onChange: (e) => setDir(e.target.value) }),
        jsx(Button, { size: 'xs', variant: 'outline', onClick: pickDir,
          children: t('yache.mdl_pick') }),
        jsx(Button, { size: 'xs', disabled: busy === 'dir' || !dir.trim(),
          onClick: () => applyDir('scan'), children: t('yache.mdl_read') }),
        lib && lib.active_dir && lib.active_dir.custom
          ? jsx(Button, { size: 'xs', variant: 'ghost',
            onClick: () => applyDir('reset'), children: t('yache.mdl_reset') })
          : null,
      ] }),
      lib && lib.scan_error
        ? jsx('p', { className: 'text-[11px] text-rose-500',
          children: t('yache.scan_err', lib.scan_error) })
        : null,
    ] }),
    !dirDirty && present.length ? jsx(Select, {
      value: vt || '',
      disabled: !!busy,
      onValueChange: (v) => {
        setVt(v || null);
        setSel(String(v || '').split('|')[0] === 'voicedesign' ? 'design' : '');
        setNote('');
      },
      children: jsxs('div', { children: [
        jsx('div', { className: 'mb-1 text-[11px] text-muted-foreground',
          children: t('yache.vt_title') }),
        jsx(SelectTrigger, { className: 'h-8 w-full text-xs',
          children: jsx(SelectValue, { placeholder: t('yache.vt_ph') }) }),
        jsx(SelectContent, { children: vtList.map((o) =>
          jsx(SelectItem, { value: o.v, children: o.label }, o.v)) }),
      ] }) }) : null,
    !dirDirty && !present.length && lib && !lib.scan_error
      ? jsx('p', { className: 'text-[11px] text-amber-500',
        children: t('yache.mdl_fresh') })
      : null,
    dirDirty ? jsx('p', { className: 'text-[11px] text-amber-500',
      children: t('yache.mdl_dirty') }) : null,
    curVt && !dirDirty ? jsxs(Fragment, { children: [
      jsx('p', { className: 'text-[10px] leading-4 text-muted-foreground',
        children: t('yache.vt_hint') }),
      missing.length ? jsx('p', { className: 'text-[11px] text-amber-500',
        children: t('yache.models_missing', missing.join(', ')) }) : null,
      lib && !lib.tokenizer_ok ? jsx('p',
        { className: 'text-[11px] text-rose-500',
          children: t('yache.tokenizer_missing') }) : null,
      curVt === 'voicedesign' ? jsx(Input, { value: instruct, disabled: !!busy,
        placeholder: t('yache.design_ph'),
        onChange: (e) => setInstruct(e.target.value),
        className: 'h-8 w-full text-xs' }) : curVt === 'base' && !options.length
        ? jsx('p', { className: 'text-[11px] text-muted-foreground',
          children: t('yache.clone_none') })
        : jsxs('div', { children: [
          jsx('div', { className: 'mb-1 text-[11px] text-muted-foreground',
                       children: t('yache.pick_voice') }),
          jsxs(Select, { value: sel, onValueChange: selChanged,
            disabled: busy || !options.length, children: [
            jsx(SelectTrigger, { className: 'h-8 w-full text-xs',
                                 children: jsx(SelectValue, {}) }),
            jsx(SelectContent, { children: options.map((o) =>
              jsx(SelectItem, { value: o.v, children: o.label }, o.v)) }),
          ] }),
        ] }),
      sel.indexOf('cln:') === 0 && lib ? (() => {
        const c = (lib.clones || []).find((x) => 'cln:' + x.name === sel);
        return c && c.description ? jsx('p', {
          className: 'text-[10px] text-muted-foreground',
          children: c.description }) : null;
      })() : null,
      sel.indexOf('cln:') === 0 ? jsx('div', {
        className: 'flex items-center', children:
        jsx(Button, { size: 'xs', variant: 'outline',
          disabled: !!busy,
          title: t('yache.dz_drop_tip'),
          onClick: () => { const n = sel.slice(4);
            if (delArm === n) { void dropClone(n); return; }
            setDelArm(n);
            setTimeout(() => setDelArm((a) => (a === n ? '' : a)), 3000); },
          children: delArm === sel.slice(4)
            ? t('yache.dz_drop_confirm') : t('yache.dz_drop') }) }) : null,
      jsxs('div', { className: 'flex items-center gap-1.5', children: [
        jsx(Input, { value: text, disabled: !!busy, placeholder: t('yache.test_ph'),
          onChange: (e) => setText(e.target.value),
          className: 'h-8 flex-1 text-xs' }),
        jsx(Button, { size: 'xs', variant: 'outline',
          disabled: !!busy || !sel || designMissing,
          onClick: () => void preview(),
          children: busy === 'prev' ? t('yache.test_busy') : t('yache.test_go') }),
        sel === 'design' ? jsx(Button, { size: 'xs',
          disabled: !!busy || designMissing,
          onClick: () => void save(),
          children: busy === 'save' ? t('yache.voice_saving') : t('yache.voice_save') }) : null,
              jsx('p', { className: 'text-[10px] leading-4 text-amber-400',
                children: t('yache.clone_next_step') }),
      ] }),
      curVt === 'voicedesign' ? jsxs('div', { className: 'space-y-1.5 border-t border-border pt-2', children: [
        jsxs('div', { className: 'flex flex-wrap items-center gap-1.5', children: [
          jsx(Button, { size: 'xs', variant: 'outline',
            disabled: !!busy || !!dzBusy || !clip,
            title: t('yache.dz_export_tip'),
            onClick: () => void exportClip(),
            children: dzBusy === 'export' ? t('yache.test_busy') : t('yache.dz_export') }),
          jsx(Input, { value: adoptName, disabled: !!busy || !!dzBusy,
            placeholder: t('yache.dz_name_ph'),
            title: t('yache.dz_adopt_tip'),
            onChange: (e) => setAdoptName(e.target.value),
            className: 'h-8 w-44 text-xs' }),
          jsx(Button, { size: 'xs',
            disabled: !!busy || !!dzBusy || !clip || !adoptName.trim(),
            onClick: () => void adoptClip(),
            children: dzBusy === 'adopt' ? t('yache.voice_saving') : t('yache.dz_adopt') }),
        ] }),
        jsx('p', { className: 'text-[10px] leading-4 text-muted-foreground',
          children: clip ? t('yache.dz_ready') : t('yache.dz_need_clip') }),
      ] }) : null,
      curVt === 'base' ? jsxs('div', { className: 'space-y-1.5', children: [
        jsx('div', { className: 'border-t border-border pt-2', children:
          jsx(Button, { size: 'xs', variant: 'outline', disabled: !!busy,
            onClick: () => { setShowNew((v) => !v); setAsr(''); },
            children: t('yache.clone_new_title') }) }),
        showNew ? jsxs('div', { className: 'space-y-1.5 rounded-lg border border-border p-2', children: [
          jsx(Input, { value: nc.name, placeholder: t('yache.clone_name_ph'),
            onChange: (e) => setNc((v) => ({ ...v, name: e.target.value })),
            className: 'h-8 w-full text-xs' }),
          jsxs('div', { className: 'flex items-center gap-1.5', children: [
            jsx(Input, { value: nc.ref_audio, placeholder: t('yache.clone_ref_ph'),
              onChange: (e) => setNc((v) => ({ ...v, ref_audio: e.target.value })),
              className: 'h-8 min-w-0 flex-1 text-xs' }),
            jsx(Button, { size: 'xs', variant: 'outline', onClick: () => void pickRef(),
              children: t('yache.clone_ref_pick') }),
          ] }),
          asr === 'busy' ? jsx('p', { className: 'text-[10px] text-muted-foreground', children: t('yache.ref_asr_busy'), key: 'asrb' }) : null,
          asr === 'ok' ? jsx('p', { className: 'text-[10px] text-emerald-500', children: t('yache.ref_asr_ok'), key: 'asro' }) : null,
          asr === 'fail' ? jsx('p', { className: 'text-[10px] text-amber-500', children: t('yache.ref_asr_fail'), key: 'asrf' }) : null,
          jsx(Input, { value: nc.ref_text, placeholder: t('yache.clone_text_ph'),
            onChange: (e) => setNc((v) => ({ ...v, ref_text: e.target.value })),
            className: 'h-8 w-full text-xs' }),
          jsx(Input, { value: nc.description, placeholder: t('yache.clone_desc_ph'),
            onChange: (e) => setNc((v) => ({ ...v, description: e.target.value })),
            className: 'h-8 w-full text-xs' }),
          jsx(Button, { size: 'xs',
            disabled: busy === 'clone' || !nc.name.trim() || !nc.ref_audio.trim(),
            onClick: () => void createClone(),
            children: busy === 'clone' ? t('yache.voice_saving') : t('yache.clone_go') }),
          jsx(Button, { size: 'xs', variant: 'outline',
            disabled: busy === 'save' || !sel || sel === 'design',
            onClick: () => void save(),
            children: busy === 'save' ? t('yache.voice_saving') : t('yache.voice_save') }),
        ] }) : null,
      ] }) : null,
    ] }) : (present.length ? jsx('p', {
      className: 'text-[11px] text-muted-foreground',
      children: t('yache.vt_pick_hint') }) : null),
    present.length && !curVt ? jsx('p', {
      className: 'text-[10px] leading-4 text-muted-foreground',
      children: t('yache.vt_hint') }) : null,
    note ? jsx('p', { className: 'whitespace-pre-wrap text-[11px] '
      + (note.startsWith(t('yache.voice_fail')) ? 'text-rose-500'
        : note.startsWith('\u2705') ? 'text-emerald-500' : 'text-muted-foreground'),
      children: note }) : null,
  ] });
}

/* One entry card per section: live dot + one-line truth pulled from the
   same doctor payload, so the home page never drifts from reality. */
function SectionCard({ t, title, summary, ok, onOpen, locked, off, extra }) {
  return jsxs('div', { className: cn(
    'flex items-center justify-between gap-3 rounded-lg border border-border bg-card p-3',
    (locked || off) && 'opacity-60'), children: [
    jsxs('div', { className: 'min-w-0 flex-1', children: [
      jsxs('div', { className: 'flex items-center gap-2', children: [
        jsx('span', { className: cn('size-2 shrink-0 rounded-full',
          off ? 'bg-sky-500'
            : locked ? 'bg-amber-500/70'
              : ok === null ? 'bg-muted-foreground/50' : ok ? 'bg-emerald-500' : 'bg-rose-500') }),
        jsx('span', { className: 'text-sm font-semibold', children: title }),
      ] }),
      jsx('div', { className: 'mt-1 truncate text-[11px] text-muted-foreground',
                   title: summary, children: summary }),
    ] }),
    extra || null,
    jsx(Button, { size: 'xs', variant: 'outline', disabled: !!locked || !!off,
                  title: off ? t('yache.dormant_tip')
                    : locked ? t('yache.sec_locked_tip') : undefined,
                  onClick: () => { if (!locked && !off) onOpen(); },
                  children: off ? t('yache.btn_dormant')
                    : locked ? t('yache.sec_locked') : t('yache.sec_open') }),
  ] });
}
function SettingsPage({ ctx }) {
  const t = usePluginI18n(ID);
  const { doc, st, err, refresh } = useDoctor(ctx, true);
  const { act, busy: actBusy, switchTo: actSwitch, refresh: refreshAct } = useActivation(ctx, true);
  const [checking, setChecking] = useState(false);
  const providerOf = (a, k) => {
    const p = a && a.providers && a.providers[k];
    return (!p || p === '(stock default)') ? '' : p;
  };
  const dormant = (a, k) => {
    const m = a && a.modules && a.modules[k];
    return !!(providerOf(a, k) && m && m.state === 'ready');
  };
  const lamps = (doc && doc.lamps) || LAMP_IDS.map((id) => ({ id, ok: false }));
  const [open, setOpen] = useState(null);
  const { lib, err: libErr, refresh: refreshLib } = useVoiceLibrary(ctx, true);
  const [ws, setWs] = useState(null);
  useEffect(() => {
    let live = true;
    ctx.rest('/wake-session', { timeoutMs: 8000 })
      .then((d) => { if (live && d) setWs(d); }).catch(() => {});
    return () => { live = false; };
  }, [ctx]);
  const [sttCur, setSttCur] = useState(null);
  useEffect(() => {
    let live = true;
    ctx.rest('/stt/state', { timeoutMs: 130000 })
      .then((d) => { if (live && d) setSttCur(d); }).catch(() => {});
    return () => { live = false; };
  }, [ctx]);
  // Section dots read the honest doctor payload: which lamps back which
  // section (TTS = engine/models/worker, wake = registration proves the
  // stock sherpa engine is mounted, cloning = voice library lamp).
  const okOf = (ids) => (doc && Array.isArray(doc.lamps)
    ? ids.every((id) => doc.lamps.some((l) => l.id === id && l.ok))
    : null);
  /* The unlock truth: a voice saved through the cloning panel — probe
     synthesized at save time. */
  const cloneReady = !!(lib && lib.saved_voice);
  const cloudProv = {
    tts: dormant(act, 'tts') ? providerOf(act, 'tts') : '',
    stt: dormant(act, 'stt') ? providerOf(act, 'stt') : '',
  };
  const secs = [
    { id: 'clone', ok: okOf(['voice']),
      summary: lib ? (lib.saved_voice
        ? voiceView(lib.saved_voice, t) : (cloudProv.tts ? t('yache.cloud_clone_detail', cloudProv.tts) : t('yache.voice_unset'))) : '—' },
    { id: 'tts', ok: okOf(['engine', 'models', 'worker']),
      locked: !cloneReady,
      summary: lib ? (cloneReady
        ? (quantLabel((lib.saved_tts && lib.saved_tts.quant) || 'auto', t)
          + (st && st.selected ? ' \u00b7 ' + st.selected : ''))
        : (cloudProv.tts ? t('yache.cloud_tts_detail', cloudProv.tts) : t('yache.sec_locked_tip')))
        : (cloudProv.tts ? t('yache.cloud_tts_detail', cloudProv.tts) : '—') },
    { id: 'wake', ok: okOf(['plugin']),
      summary: ws ? (ws.start_new_session ? t('yache.wake_opt_new') : t('yache.wake_opt_keep')) : '—' },
    { id: 'stt', ok: okOf(['plugin']),
      summary: sttCur
        ? (sttCur.engine_ok
          ? sttCur.model + ' \u00b7 '
            + (sttCur.device === 'auto' ? t('yache.stt_dev_auto')
                                        : String(sttCur.device).toUpperCase())
          : (cloudProv.stt ? t('yache.cloud_stt_detail', cloudProv.stt) : t('yache.stt_engine_down')))
        : (cloudProv.stt ? t('yache.cloud_stt_detail', cloudProv.stt) : '—') },
  ];
  /* Cloud honesty: when the TTS/STT provider slot is held by another
     provider, the card shows a dormant dot, a truthful summary, a
     disabled-open button and a one-click SwitchBack — no engine, no
     residency, and the official entry point keeps full control.  Wake is
     independent by design and never goes gray; the clone card follows
     the TTS provider because cloned voices only ever ride our engine. */
  for (const s of secs) {
    s.off = false; s.extra = null;
    if ((s.id === 'tts' || s.id === 'clone') && cloudProv.tts) {
      s.off = true;
      if (s.id === 'tts') s.extra = jsx(Button, { size: 'xs', variant: 'outline',
        disabled: !!actBusy, title: t('yache.switching'),
        onClick: () => actSwitch('tts'),
        children: actBusy === 'tts' ? t('yache.switching') : t('yache.switch_back') });
    } else if (s.id === 'stt' && cloudProv.stt) {
      s.off = true;
      s.extra = jsx(Button, { size: 'xs', variant: 'outline',
        disabled: !!actBusy, title: t('yache.switching'),
        onClick: () => actSwitch('stt'),
        children: actBusy === 'stt' ? t('yache.switching') : t('yache.switch_back') });
    } else if (s.id === 'wake' && ws && !ws.start_new_session) {
      s.summary = t('yache.wake_off_card');
    }
  }
  const bodies = {
    tts: () => jsx(TtsBody, { ctx, t, st, lib, refreshLib }),
    stt: () => jsx(SttBody, { ctx, t }),
    wake: () => jsx(WakeBody, { ctx, t, st }),
    clone: () => jsx(CloneBody, { ctx, t, lib, refreshLib }),
  };
  return jsxs('div', { className: 'mx-auto max-w-3xl space-y-4 p-6', children: [
    jsxs('div', { className: 'flex items-center justify-between', children: [
      jsx(Button, { size: 'xs', variant: 'outline',
                    onClick: goBack,
                    children: t('yache.back') }),
      /* The doctor runs in milliseconds and the lamps rarely change, so a
         silent refresh looks like a dead button.  Spin the label while the
         round-trip is in flight and toast the outcome when it lands —
         the user clicked, the screen must answer. */
      jsx(Button, { size: 'xs', variant: 'outline', disabled: !!checking,
                    onClick: async () => {
                      setChecking(true);
                      let bad = false;
                      try { await Promise.all([refresh(), refreshAct()]); }
                      catch { bad = true; }
                      setChecking(false);
                      try {
                        host.notify({ kind: bad ? 'warning' : 'success',
                          message: bad ? t('yache.refresh_failed') : t('yache.refresh_done') });
                      } catch { /* best-effort toast */ }
                    },
                    children: checking ? t('yache.refreshing') : t('yache.refresh') }),
    // Log button — the support trail lives in the plugin's own log (host
    // logs only show the one-line summary). Reveals it in Explorer with
    // the file selected; silent by design, the window appearing IS the
    // confirmation.
    jsx(Button, { variant: 'outline', size: 'sm',
      style: { pointerEvents: 'auto' },
      onClick: async () => {
        try {
          const r = await ctx.rest('/debug/open_log', { method: 'POST' });
          if (r && r.ok === false) {
            try {
              host.notify({ kind: 'warning',
                message: t('yache.open_log_fail')
                  + String(r.error || 'failed') });
            } catch { /* best-effort toast */ }
          }
        } catch { /* opener trouble must never bite the page */ }
      },
      children: t('yache.open_log') }),
    ] }),
    jsx('div', { children: jsxs('h2', { className: 'text-lg font-semibold', children: [
      t('yache.page_title'),
    ] }) }),
    jsx('p', { className: 'text-xs text-muted-foreground', children: t('yache.page_desc') }),
    (err || libErr) ? jsx('div', { className: 'text-xs text-rose-500', children: err || libErr }) : null,
    jsx('div', { className: 'rounded-lg border border-border bg-card p-3', children:
      jsxs(Fragment, { children: [
        jsx('div', { className: 'mb-1 text-xs font-semibold', children: t('yache.doctor') }),
        jsx(Fragment, { children: lamps.map((l) =>
          jsx(LampRow, { t, lamp: l, full: true }, 'p-' + l.id)) }),
        st ? jsxs('div', { className: 'mt-1 border-t border-border pt-1', children: [
          StatusRow({ label: st.has_nvidia ? t('yache.vram') : t('yache.vram_none'),
            value: st.has_nvidia && st.free_vram_mib != null ? st.free_vram_mib + ' MiB' : '\u2014' }),
          st.recommended ? StatusRow({ label: t('yache.rec'),
            value: st.recommended.tier, title: st.recommended.reason || '' }) : null,
        ] }) : null,
      ] }) }),
    jsx('div', { className: 'grid gap-3 sm:grid-cols-2', children: secs.map((s) =>
      jsx(SectionCard, { key: s.id, t,
        title: t('yache.sec_' + s.id), summary: s.summary, ok: s.ok,
        locked: !!s.locked,
        off: !!s.off, extra: s.extra,
        onOpen: () => { if (!s.locked && !s.off) setOpen(s.id); } })) }),
    open ? jsx(ModalShell, { t,
      title: t('yache.sec_' + open), onClose: () => setOpen(null),
      children: bodies[open]() }) : null,
  ] });
}

export default {
  id: ID,
  name: 'Yache Voice',
  description: 'Status-bar console for the Yache local voice pipeline (TTS/STT/wake word).',
  register(ctx) {
    ctx.i18n?.register?.(LOCALES);
    // Feed the file log pipe + catch everything that escapes React (see dbg).
    _api = ctx;
    hookGlobalErrorNet();
    dbg('panel mounted');
    const chipText = (key, fb) => {
      try {
        const s = ctx.i18n?.t?.(key);
        return typeof s === 'string' && s && !s.includes('{{') ? s : fb;
      } catch { return fb; }
    };
    ctx.register({
      id: ID + ':chip',
      area: STATUSBAR_AREAS.right,
      order: 40,
      // Declarative (hermes-wallpaper shape): host owns the trigger button.
      data: {
        id: ID + ':chip',
        icon: jsx(ChipIcon, { ctx }),
        title: chipText('yache.brand', 'Yache Voice'),
        toggleLabel: chipText('yache.brand', 'Yache Voice'),
        variant: 'menu',
        menuAlign: 'end',
        menuClassName: 'w-auto',
        menuContent: (close) => jsx(PanelMenu, { ctx, close }),
      },
    });
    ctx.registerSettingsPage?.({
      id: ID,
      // Follows the host language (zh: 雅澈语音 / en: Yache Voice) and
      // carries the SAME icon the statusbar chip uses, so the plugin
      // hub menu, the statusbar and the settings rail all agree.
      title: chipText('yache.brand', 'Yache Voice'),
      icon: 'comment-discussion',
      render: () => jsx(SettingsPage, { ctx }),
    });
    ctx.register?.({
      id: 'yache-doctor',
      area: PALETTE_AREA,
      order: 40,
      data: {
        id: 'yache-doctor',
        label: '雅澈语音体检 / Yache Voice doctor',
        keywords: ['voice', 'tts', 'stt', '语音', '体检', '雅澈'],
        run: () => navigate(pluginSettingsHref(ID)),
      },
    });
  },
};
