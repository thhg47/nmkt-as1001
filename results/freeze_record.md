# Legacy validation draft — not an approved freeze

This file was generated during development before the group formally approved a freeze. Its candidate metrics are historical and its model configuration is incomplete. Do not submit or use it for final evaluation.

The legacy model artifact in this directory is not accepted by `analysis.py --stage final`. To create the required root-level freeze record and model artifact, review validation results with the group and explicitly run:

```bash
python analysis.py --stage freeze --approve-freeze
```

The final evaluation stage requires the approved root-level `freeze_record.json` and refuses to overwrite an existing final output.
