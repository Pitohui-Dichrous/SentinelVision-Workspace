# SentinelVision UI design system

## Direction and source

This design applies the user-requested [UI UX Pro Max skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) to the existing PySide6 desktop application. It follows the skill's verified `industrial safety workspace minimal` design-system result: **Minimalism & Swiss Style**, variance 2, motion 2, density 3. The user's Apple website direction takes priority over the database's industrial orange palette: warm white, charcoal typography and a single blue action color.

The implementation stays native and offline. PySide6 is not one of the skill's supported stack searches, so Qt implementation choices use the repository's existing widgets. The reduced-motion UX search matched the skill's Reduced Motion guidance. The progressive-disclosure search returned an unrelated heading rule even after a retry; form disclosure instead follows the skill's built-in quick reference, not that search result. No GSAP or web runtime is added.

## Visual rules

- Background `#F5F5F7`, surface `#FFFFFF`, text `#1D1D1F`, secondary text `#626269`, action `#0066CC`. `ui_theme.py` supplies the matching dark theme and all semantic status colors.
- Layout creates hierarchy through whitespace and type. Use flat surfaces, restrained separators and 20px section corners; avoid nested bordered cards, decorative status colors or looping motion.
- Main content is centered at approximately 1180px on wide windows. Spacing follows 8, 12, 16, 24, 32px steps. Form labels remain visible.
- Type: Segoe UI with Microsoft YaHei UI fallback on Windows. Offscreen rendering loads the installed static Chinese font; the bundled Noto font remains the portable fallback. Hero 54px, page title 36px, section title 20px, body 14px, caption 12px.
- Small native SVG chevrons and check marks, plus a static Qt vector optical motif. The motif is decorative; it represents no live camera, measurement or model output.
- Normal text, semantic badges and action labels target at least 4.5:1 contrast. Focus borders stay visible; each interactive control retains native keyboard behavior.

## Information architecture

1. **开始**: one primary action, start detection. Show computing device, deployed model count and candidate count. Keep training, review and the model library one step away. Maintenance lives in a menu.
2. **训练模型**: dataset, foundation model, task name and epochs first. Custom weights and advanced parameters expand on demand. Preserve the ten official weight choices, safe resume, audit, stop and log output.
3. **审核与部署**: evidence on the left and four explicit review checks on the right. A readable class summary opens a separate editor for each class. Media-test output stays on this page. Candidate changes clear every approval. The final deployment dialog remains mandatory.
4. **模型库**: current production inventory, readable state and version paths. Refresh, production folder and archive remain accessible.
5. **检测台**: video and camera controls above the image; source, model configuration and event review in the inspector. Region tools expand below playback. Display, motion, language and role controls share a preferences menu. Empty video state is explicit; real frames use a neutral dark image stage.

## Motion and layout

- Page change: 160ms OutCubic opacity 0.90 → 1.0. Input and navigation remain active; the newest navigation wins.
- Pointer press: 0.975 scale, 140ms; release 90ms. Only the painted button changes, never its layout bounds. Disable/hide resets the effect. Keyboard activation stays immediate.
- Event drawer: 220ms enter, 160ms exit; interrupted transitions start from the current position. Focus moves to the close action and returns to its origin.
- “减少动效” in the workbench and the detector motion preference use the existing `SENTINEL_REDUCE_MOTION` switch. It is a per-process preference, inherited by a launched detector, not a new persisted detection setting. Essential state changes remain immediate.
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

`tests/test_ui_redesign.py` checks form state, class edits, approval reset and final confirmation, reduced motion, disabled press feedback, small-window geometry, Viewer permissions, explicit model selection, frame rendering and contrast. UI smoke screenshots deliberately skip model inspection and inference; their empty table is not evidence of an empty production directory. Native/offscreen UI checks do not replace real-media detection validation.
