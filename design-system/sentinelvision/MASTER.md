# SentinelVision UI design system

## Direction and source

This design applies the user-requested [UI UX Pro Max skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) to the existing PySide6 desktop application. It follows the skill's verified `industrial safety workspace minimal` design-system result: **Minimalism & Swiss Style**, variance 2, motion 2, density 3. The user's Apple website direction takes priority over the database's industrial orange palette: warm white, charcoal typography and a single blue action color.

The implementation stays native and offline. PySide6 is not one of the skill's supported stack searches, so Qt implementation choices use the repository's existing widgets. Motion follows the local `animate` skill; the user's explicit request to omit reduced-motion support takes priority over that skill's default. The progressive-disclosure search returned an unrelated heading rule even after a retry; form disclosure instead follows the skill's built-in quick reference, not that search result. No GSAP or web runtime is added.

## Visual rules

- Background `#F5F5F7`, surface `#FFFFFF`, text `#1D1D1F`, secondary text `#626269`, action `#0066CC`. `ui_theme.py` supplies the matching dark theme and all semantic status colors.
- Layout creates hierarchy through whitespace and type. Use flat surfaces, restrained separators and 20px section corners; avoid nested bordered cards, decorative status colors or looping motion.
- Main content is centered at approximately 1180px on wide windows. Spacing follows 8, 12, 16, 24, 32px steps. Form labels remain visible.
- Type: Segoe UI with Microsoft YaHei UI fallback on Windows. Offscreen rendering loads the installed static Chinese font; the bundled Noto font remains the portable fallback. Hero 54px, page title 36px, section title 20px, body 14px, caption 12px.
- Small native SVG chevrons and check marks, plus a static Qt vector optical motif. The motif is decorative; it represents no live camera, measurement or model output.
- Normal text, semantic badges and action labels target at least 4.5:1 contrast. Focus borders stay visible; each interactive control retains native keyboard behavior.
- Comboboxes use Qt's list popup (`combobox-popup: 0`) so the themed list owns its border and height. The menu-style container otherwise exposes the system palette as black gutters behind rounded lists and can clip short menus. This switch is supported by the bundled [Qt 6.6.3 stylesheet implementation](https://github.com/qt/qtbase/blob/v6.6.3/src/widgets/styles/qstylesheetstyle.cpp#L602). Keep native scrolling, selection, keyboard confirmation and Escape behavior; verify short/long menus in both themes and at 150% scaling.

## Information architecture

1. **开始**: one primary action, start detection. Show computing device, deployed model count and candidate count. Keep training, review and the model library one step away. Maintenance lives in a menu.
2. **训练模型**: dataset, foundation model, task name and epochs first. Custom weights and advanced parameters expand on demand. Preserve the ten official weight choices, safe resume, audit, stop and log output.
3. **审核与部署**: evidence on the left and four explicit review checks on the right. A readable class summary opens a separate editor for each class. Media-test output stays on this page. Candidate changes clear every approval. The final deployment dialog remains mandatory.
4. **模型库**: current production inventory, readable state and version paths. Refresh, production folder and archive remain accessible.
5. **检测台**: video and camera controls above the image; source, model configuration and event review in the inspector. Region tools expand below playback. Display, language and role controls share a preferences menu. Empty video state is explicit; real frames use a neutral dark image stage.

## Motion and layout

- `ui_motion.py` owns shared duration and curve tokens. Page and press easing is `cubic-bezier(0.23, 1, 0.32, 1)`; drawer easing is `cubic-bezier(0.32, 0.72, 0, 1)`.
- Page change: a 12px upward entrance over 240ms, with a 120ms crossfade and a 3px outgoing drift. Text resolves before the movement settles. Header and content move together; navigation stays stable. A mouse-transparent snapshot overlay paints the transition without moving live controls or delaying state changes. Snapshots include the actual window background to prevent color flashes. Rapid navigation starts from the visible composite. Resize/hide/style changes cancel it and immediately release both snapshots; normal completion does the same.
- Inspector tabs: pointer changes use the same transition at 180ms and 8px. Keyboard and programmatic tab changes are immediate. Live video and incoming detection data never animate.
- Pointer press: 0.97 scale, 120ms; release 180ms. A tap released between frames gets a brief 60ms press to 0.98 before release, while its action fires immediately. Reversals begin at the current painted size. Only the painted button changes, never layout or hit bounds. Disable/hide restores its full size. Keyboard activation stays immediate, including navigation triggered through a button.
- Event drawer: 260ms enter, 200ms exit; interrupted transitions start from the current position. Focus moves to the close action and returns to its origin. Escape closes immediately; resizing snaps to the correct open or closed position.
- Per the user's request, there is no reduced-motion control or environment switch. Existing detection settings are unaffected.
- Workbench minimum 960×640. Below 1200px training and review stack vertically; below 1100px the illustration and low-priority model-path column collapse. Pages scroll vertically while navigation stays available.
- Detector minimum 1100×720. Short windows reduce the image minimum height so playback and expanded region controls never overlap the image.

## Preserved boundaries

Training writes to `TRAINING_OUTPUTS`. Production models remain manually approved in `RESULTS`. A new model stays unchecked until the user chooses it. Detection class mapping, overlap protection, legacy pipeline configuration, temporal evidence processing, alerts and review journals retain their existing behavior. The redesign does not approve or deploy any model.

## Verification

Use the portable Python runtime with `-I -B -X utf8`:

```text
RUNTIME/python/python.exe -I -B -X utf8 -m unittest discover -s tests -q
RUNTIME/python/python.exe -I -B -X utf8 portable/ui_smoke.py --screenshots .runtime/ui-redesign/wide --width 1440 --height 900
RUNTIME/python/python.exe -I -B -X utf8 portable/ui_smoke.py --screenshots .runtime/ui-redesign/compact --width 960 --height 640
RUNTIME/python/python.exe -I -B -X utf8 portable/verify_workspace.py
RUNTIME/python/python.exe -I -B -X utf8 portable_check.py
```

`tests/test_ui_redesign.py` checks form state, class edits, approval reset and final confirmation, interruptible page/tab/drawer transitions, snapshot cleanup, rapid taps, pointer cancellation, keyboard activation, disabled press feedback, small-window geometry, Viewer permissions, explicit model selection, frame rendering and contrast. UI smoke screenshots deliberately skip model inspection and inference; their empty table is not evidence of an empty production directory. Native/offscreen UI checks do not replace real-media detection validation.
