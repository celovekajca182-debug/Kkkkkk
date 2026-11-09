# Third-party asset provenance

## Recorded audio

The game currently plays recorded CC0 samples from the public audio directory of [yegors/hard-lines](https://github.com/yegors/hard-lines/tree/main/public/audio):

- HK416 report recordings are used for M416 and SCAR-L.
- AK-47 report recordings are used for AKM.
- AWM report recordings are used for Kar98k.
- Recorded hard-surface footstep takes are used for player movement.

The source project documents the firearm recordings as edits of genuine recordings from The Free Firearm Sound Library, CC0 1.0, and the hard-surface footsteps as Joseph Sardin / BigSoundBank CC0 recordings. See the [source audio notes](https://github.com/yegors/hard-lines/blob/main/public/audio/README.md).

Audio is loaded from raw GitHub URLs at runtime; an internet connection is required for these samples. The current implementation intentionally does not synthesize replacement gunshots when samples fail to load.

## 3D content

The transport aircraft added in this branch is a Three.js mesh assembly, not an imported aircraft asset. Existing characters, environment props, weapons, and buildings also remain runtime geometry. No third-party model files are currently bundled.
