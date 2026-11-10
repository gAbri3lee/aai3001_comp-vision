# Manual test sheet

These checks need a person, a real camera, speakers or a Roboflow account.
Automated tests cannot do them. Tick each box and add a note with the date.

Start the app with `python src/webapp.py --port 5001 --no-depth` and open
`http://127.0.0.1:5001` (add `/?debug=1` to see ignored boxes and sizes).

- [ ] Gabriel's latest push has been pulled and reviewed before any work starts.
- [ ] Live webcam works at the default camera index. (macOS may ask for Terminal
      camera permission.)
- [ ] Spoken alerts are audible, understandable and not repeated too often while walking.
- [ ] A person in the background does not trigger speech; the same person up close does.
- [ ] A person at the far side edge is ignored; a near one is spoken as "on your left"
      or "on your right".
- [ ] A pillar is sometimes missed. This is expected (weak class), not a bug.
- [ ] A chair or table in the path gives "Object detected ahead"; one at the side
      or far away is silent.
- [ ] Banner text, highlighted (magenta) box and spoken phrase match.
- [ ] Local and cloud give similar behaviour on the same walk. Note the cloud delay.
- [ ] Tape-measure distance test completed and results saved in
      `docs/DISTANCE_TEST_SHEET.md`.
- [ ] Gate sizes reviewed against the distance test and the Phase D proposal in
      `docs/GUI_TEST_REPORT.md`; `alert_gate.calibrated` set to true only after that.
- [ ] Cloud endpoint tested from a second device (needs a real hosted URL, not localhost).
- [ ] The old Roboflow key has been revoked in Roboflow and a new private key
      generated, then stored only as the `ROBOFLOW_API_KEY` environment variable or
      a Kaggle secret. This cannot be done from code.

Notes:

| Date | Tester | Check | Result |
|---|---|---|---|
| | | | |
