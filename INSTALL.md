# Installing OpticalFlake

## macOS

1. Unzip the download in Finder.
2. Drag `OpticalFlake_V0.5.0.app` to **Applications**.
3. Unblock it (macOS will otherwise say the app is "damaged" or "cannot be opened"):

   ```bash
   sudo xattr -dr com.apple.quarantine /Applications/OpticalFlake_V0.5.0.app
   ```
(sudo may not be strictly necessary)
4. Open it.
5. Allow screen capture: **System Settings → Privacy & Security → Screen Recording** → add
   OpticalFlake, then quit and reopen the app.

The following may work depending on OS version:
No Terminal — open the app, dismiss the warning, then **System Settings → Privacy &
Security**, scroll down, click **Open Anyway**. (Right-click → Open no longer works on
macOS 15+.)

`can't be opened` even after all of the above — check the chip with `uname -m`. This build is
Apple Silicon only; `x86_64` means an Intel Mac, which needs a separate build.

## Windows

1. Download `OpticalFlake_V0.5.0.exe`.
2. Run it. On the SmartScreen warning, click **More info → Run anyway**.

## Notes

- Repeat the unblock and the Screen Recording step for every new version — macOS ties both to
  the exact app bundle.
- Send builds as a `.zip`. Uploading the unpacked `.app` to Drive or Dropbox breaks it.
- The warnings are expected: this is an internal lab tool, not signed with a paid Apple
  Developer ID.
