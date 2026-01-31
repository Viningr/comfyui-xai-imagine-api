import type { ComfyApp } from "@comfyorg/comfyui-frontend-types";
import { SETTINGS_IDS } from "./constants";

declare global {
  const app: ComfyApp;

  interface Window {
    app: ComfyApp;
  }
}

app.registerExtension({
  name: "ComfyUI Xai Imagine Api Node",
  settings: [
    {
      id: SETTINGS_IDS.VERSION,
      name: "Version 1.0.0",
      type: () => {
        const spanEl = document.createElement("span");
        spanEl.insertAdjacentHTML(
          "beforeend",
          `<a href="https://github.com/Viningr/comfyui-xai-imagine-api" target="_blank" style="padding-right: 12px;">Homepage</a>`
        );

        return spanEl;
      },
      defaultValue: undefined,
    },
    {
      id: SETTINGS_IDS.DEBUG_LOGGING,
      name: "Enable Debug Logging",
      type: "boolean",
      tooltip:
        "Show detailed debug logs in browser console during operation",
      defaultValue: false,
    },
  ]
});
