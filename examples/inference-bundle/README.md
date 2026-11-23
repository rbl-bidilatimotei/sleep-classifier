# Contract examples only

manifest.example.json illustrates packaging metadata for the seed-42 fold-01 research reference.
It is not an assembled inference bundle. Its referenced files are not present here.
The runtime revision identifies the merged source snapshot, not a new training run.
Packaged-file digests remain placeholders until an exporter produces the files.
All packaged-file digests are explicit zero placeholders until the exporter assembles and hashes the files.
The original verified checkpoint digest appears separately in provenance.source_checkpoint_sha256.
The runtime revision names the merged inference implementation; no bundle has been exported.

The request and response contain synthetic values. The response is not a measured prediction for that request.
Use the versioned schemas to validate structure. Do not load or deploy this directory as a bundle.
See the [contract](../../docs/inference-bundle-contract.md) for semantic checks that schema validation cannot perform.
