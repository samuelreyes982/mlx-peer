# MLX Peer privacy policy

Effective September 23, 2026. Developer: Samuel Reyes.

MLX Peer connects your iPhone to your Apple Silicon Mac over USB so they can run a supported local AI model together. No MLX Peer account is required.

## Data collection

MLX Peer does not collect your personal information on a developer-operated server. The iPhone app includes no advertising, tracking, analytics SDKs, cloud inference, or automatic uploads to the developer. Its privacy label is intended to describe this as **Data Not Collected**.

## Data on your devices

The iPhone stores a randomly generated pairing identifier and credential, model configuration, and the model weights assigned by your Mac. Pairing credentials are protected with iOS file protection and restricted file permissions. The pairing and model storage directories are excluded from iCloud backup.

During inference, intermediate numerical model results pass between your Mac and iPhone over a local USB connection. These results can contain information derived from your prompts and should be treated as sensitive. Pair only with your own trusted Mac. The application authenticates the paired Mac but does not add end-to-end encryption to the USB transport. It does not provide Wi-Fi or Internet inference connections.

The Mac companion reads the model folder you select and stores its own prepared model cache and pairing credentials locally. Prompts and responses are handled on your own devices; the developer cannot read them through MLX Peer. Device memory and available storage are checked to avoid loading a model that the phone cannot safely accommodate. File metadata is used to validate local model files and resume interrupted transfers. This information is not used for advertising or fingerprinting.

## Retention and deletion

Model files and pairing credentials remain on the iPhone until you remove them or delete the app. Tap **Stop sharing**, then **Pair a new Mac** to revoke the existing pairing, or **Remove saved models** to delete the stored model weights. Deleting the iPhone app removes its local app data. Backgrounding or locking the phone stops sharing. Removing iPhone data does not delete your Mac's original model folders or caches; those are managed separately on the Mac.

## External services

Links to GitHub, model providers, or other websites open external services governed by their own privacy policies. Downloading a model separately may disclose normal network information to that provider. MLX Peer does not automatically download a model in the iPhone app. Apple may collect diagnostics depending on your device and App Store/TestFlight settings; Apple's privacy practices govern that collection.

If you voluntarily open a GitHub support issue, the issue and any attached information are public. Do not post prompts, pairing codes, credentials, personal files, or private model data. The developer uses the information you choose to provide to respond to your issue. You can edit or delete your own GitHub content using GitHub's controls, subject to GitHub's retention policies.

## Contact and updates

For support or a privacy question, use [the MLX Peer support page](SUPPORT.md) or [the project's issue tracker](https://github.com/samuelreyes982/mlx-peer/issues). Describe the question without including personal or sensitive information. Changes to this policy will be published here with an updated effective date.
