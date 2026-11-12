SOURCE AUDIT: FMAP / SIDOWN
==========================
Paper: https://arxiv.org/abs/2504.03463 (v2, 25 April 2025)
Code: https://github.com/davidlandry93/sidown
Inspected revision: 5396f9f5ecb673394b532c80a3e4922d842d1c78
The upstream checkout was read under /tmp only; it is NOT vendored here.
No LICENSE/COPYING file was found at this revision. Do not silently redistribute
the authors' source as though it had an MIT license.

WHAT THE PAPER AND SOURCE AGREE ON
---------------------------------
* 122 stations: Austria, Belgium, France, Germany, Netherlands; no Swiss stations.
* t2m and p10fg6 (6-hour maximum wind gust); log1p applied to gust for modeling.
* 20 positive lead times, 6 to 120 hours; remove the zero-hour lead.
* Reforecast years 2003, 2010, 2016 form validation; remaining reforecast years
  form training. Operational forecasts from 2017-04-01 through 2018 form test.
* Station tokens, learned station embeddings, four self-attention blocks,
  a register token, and joint station/variable outputs.
* Straight conditional flow interpolation, velocity regression, Euler sampling.
* Forecast conditioning uses ensemble summaries rather than each raw member.
* Missing target coordinates are removed from the loss, not entire forecast cases.
* LVS uses five nearby OTHER stations and p=.5 with ordered-pair sums. The joint
  variant flattens station-variable pairs inside each local neighborhood.

IMPORTANT DISCREPANCIES / IMPLEMENTATION DETAILS
------------------------------------------------
1. Paper Table 5.3/5.4: FMAP max LR 1e-4, 80 epochs. Pinned eupp_fm.yaml:
   LR 5e-5, 60 epochs. Do not describe running the YAML as exact paper replication.
2. Pinned batch2xs/ensmean.yaml selects x0_key=model_output; StartFromEnsembleMean
   currently uses batch['forecast'] regardless of that key. README mentions
   MBM-output variants. Specify the source of the baseline explicitly. Ours
   always uses the raw ensemble mean in its model-climatology anomaly space.
3. Paper describes a fitted lead-error scaling. Pinned torchdataset.py instead
   uses four manually chosen constants in a linear-in-lead formula. Ours fits
   residual RMS on the training partition and records the coefficients.
4. EulerSolver's integer argument becomes that many TIME POINTS and therefore
   one fewer updates. Our --steps means exactly the number of network calls.
5. Upstream climatology tasks depend on the full reforecast data before the
   later train/validation filtering. Ours fits all normalization and error-scale
   coefficients on training years only. This differs from those released tasks.
6. Real Zarr metadata contains 209 base initialization dates x 20 relative years.
   Source reconstructs actual years using relative_year + 1996; our reader
   implements that mapping with identity/time/lead alignment checks. The paper's
   description of the full reforecast date span should not be used as a parser.
7. Source conditioning includes missing-observation masks, and inference zeros
   them. Ours never conditions on future-observation availability; masks are
   used only in losses/evaluation. Availability fractions for INPUT predictors
   are permitted and are recorded as conditioning features.
8. Source contains a workaround for a malformed Netherlands 500-hPa operational
   store. Our independent reader includes the same narrowly scoped repair using
   surface-store coordinates and exact shape checks, without editing raw files.
   Full-profile preparation still needs checking on the complete downloaded release.
9. A real public fragment confirms 4,180 reforecast initializations in 1997-2016,
   with 3,553 training and 627 validation initializations. Operational data has
   730 initializations in 2017-2018; the April-2017 cutoff leaves 640. At all 20
   leads, expected counts before excluding fully missing cases are 71,060 / 12,540 /
   12,800. Pinned dataset YAML still lists 14,600 test examples before that cutoff.

OUR EXPERIMENT IS A CONTROLLED ADAPTATION
---------------------------------------
* Pilot: 122 stations, temperature only, lead 48 h; five predictors (t2m,u10,v10,
  tcc,tcwv), mean/std/availability summaries, station and date metadata. Width 256.
* Full profile: both variables and all leads, more predictors and width 1024;
  a second-stage configuration, not a previously validated forecast result.
* One shared independent PyTorch Transformer implementation for all four arms;
  token-wise LayerNorm/PyTorch attention differ from the authors' custom blocks.
* Endpoint models output a scaled correction; FM model outputs velocity.
* All arms select checkpoints using fixed-noise empirical validation ES. The
  released FM training task validates velocity regression instead.
* Pilot ES uses the U-statistic M(M-1) repulsion denominator, matching our earlier
  weather experiment. Full configuration uses the source baseline's M^2 estimator.
  These choices are explicit. They are not interchangeable when training M is small.
* Main comparisons use one model per seed and 51 final scenarios. Scores are
  averaged across seeds; the source ESGM paper baseline uses ten trained models.
* Temperatures are a frozen starting point, not claimed optimal on EUPPBench.
* Riesz field scales are local to a microbatch/rank, like the copied routine.
  Gradient accumulation does not merge those statistics across all 64 cases.
  Keep GPU count/microbatch size fixed when comparing objectives; hardware runs
  have separate manifests. No exact one-/four-GPU equivalence is claimed.
* We report both physical-unit and anomaly scores. The raw ensemble comparator
  is uncorrected, while the paper's first comparator is debiased IFS.
* Missing-value scores use available coordinates/pairs; metrics cannot certify
  the full distribution. No independent-replicate or significance claim.

DATA PROVENANCE
---------------
Official station archive: https://zenodo.org/records/7708362
URL: https://zenodo.org/api/records/7708362/files/EUPPBench-stations.zip/content
Size: 18,324,361,137 bytes
MD5: e409457279b3494d18f2dfb41f3f449b
Total expanded ZIP entries: 18,286,412,657 bytes, before prepared arrays.
No full archive download or full-scale training was performed locally.
The downloader validates the complete archive checksum before safe extraction.
The data's terms/attribution remain those of the original Zenodo record.

FOLLOW-UP BEFORE PAPER-LEVEL CLAIMS
----------------------------------
Verify the complete real-data preparation on the target machine; audit counts,
station identities, physical units, missing masks and raw baseline scores.
Reproduce a stronger original baseline or document every adaptation above.
Tune with a declared validation budget; evaluate the fixed recipe on test once.
Use matched ensemble sizes; report trained-model counts, parameter counts and
measured inference cost (including repeated flow calls), not just call-count ratios.
