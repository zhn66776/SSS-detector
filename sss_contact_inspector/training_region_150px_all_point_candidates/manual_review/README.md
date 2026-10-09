# Manual Costa Head patch review

A local browser interface for reviewing the patches in `training_region_150px_all_point_candidates`.

- The default view shows the raw grayscale patch; a marked preview is also available.
- Selecting the keep option and advancing records `keep`; advancing without it records `drop`.
- Decisions are saved to local JSON and CSV files, together with selected/rejected contact lists and a review summary.
- Unreviewed patches are excluded from both the selected and rejected lists.

From the repository root, with the candidate dataset restored:

```sh
cd sss_contact_inspector/training_region_150px_all_point_candidates
python3 manual_review/review_server.py --port 8787
```

Open `http://127.0.0.1:8787/`. Images, decisions and generated summaries are excluded from Git.
