"""Visual styling and theme setup for the Gradio demo."""

from __future__ import annotations

import gradio as gr


EMIL_CSS = """
/* Layout Container */
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 16px !important;
}

/* Header */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color-primary, #e4e4e7);
    margin-bottom: 24px;
}

.dark .app-header {
    border-bottom-color: #27272a;
}

.app-title-group {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.app-title {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin: 0 !important;
    color: var(--body-text-color, #09090b);
}

.app-subtitle {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued, #71717a);
}

.app-device-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.775rem;
    font-family: var(--font-mono);
    color: var(--body-text-color-subdued, #71717a);
    background: var(--background-fill-secondary, #f4f4f5);
    padding: 4px 10px;
    border-radius: 9999px;
    border: 1px solid var(--border-color-primary, #e4e4e7);
}

.dark .app-device-badge {
    background: #18181b;
    border-color: #27272a;
}

.device-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
}

.device-dot-cpu {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #a1a1aa;
}

/* Primary Button: Emil Kowalski physical response */
button.primary {
    background: #18181b !important;
    color: #fafafa !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: -0.01em !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transition: transform 120ms cubic-bezier(0.23, 1, 0.32, 1), background-color 120ms ease !important;
}

.dark button.primary {
    background: #f4f4f5 !important;
    color: #09090b !important;
    border-color: rgba(0, 0, 0, 0.1) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
}

button.primary:hover {
    background: #27272a !important;
}

.dark button.primary:hover {
    background: #e4e4e7 !important;
}

button.primary:active {
    transform: scale(0.975) !important;
}

/* Form & Input Polish */
.gr-form, .gr-box {
    border-radius: 8px !important;
    border: 1px solid var(--border-color-primary, #e4e4e7) !important;
}

.dark .gr-form, .dark .gr-box {
    border-color: #27272a !important;
}

/* Segmented / Radio */
.gr-radio {
    gap: 4px !important;
}

/* Clean Image Slider Frame */
.image-slider {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-color-primary, #e4e4e7) !important;
}

.dark .image-slider {
    border-color: #27272a !important;
}

/* Table Polish */
table {
    font-size: 0.825rem !important;
    border-collapse: collapse !important;
}

th {
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: var(--body-text-color-subdued, #71717a) !important;
    border-bottom: 1px solid var(--border-color-primary, #e4e4e7) !important;
    padding: 8px 12px !important;
}

.dark th {
    border-bottom-color: #27272a !important;
}

td {
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border-color-primary, #f4f4f5) !important;
    font-family: var(--font-mono) !important;
}

.dark td {
    border-bottom-color: #18181b !important;
}

td:first-child, td:last-child {
    font-family: var(--font-sans) !important;
}

/* Secondary Drawer Accordions */
.gr-accordion {
    border: 1px solid var(--border-color-primary, #e4e4e7) !important;
    border-radius: 8px !important;
    margin-top: 12px !important;
}

.dark .gr-accordion {
    border-color: #27272a !important;
}

.gr-accordion > .label-wrap {
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
"""


def build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        button_primary_background_fill="#18181b",
        button_primary_background_fill_hover="#27272a",
        button_primary_text_color="#fafafa",
        block_border_width="1px",
        block_radius="8px",
    )
