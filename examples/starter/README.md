# Starter project

A minimal app protected by Immune, laid out the way a real repository would be. The
[setup guide](../../docs/guides/setup.md) explains every file.

```bash
cd examples/starter
pip install -r requirements.txt
immune config validate immune.yaml
immune vaccines test
immune replay
pytest
```

Everything above runs offline. `app.py` needs `OPENAI_API_KEY` and `TYPESAFE_API_KEY` to answer real questions.
