# 008: Dataset Size as Primary Improvement Bottleneck

**Status:** Open (ongoing data collection)
**Severity:** High
**Area:** Data Collection, ML Performance

## Problem

With 136 events and 42 artists (up from 81/23 at time of writing), the dataset has grown substantially but remains a constraint on model improvement. Many attempted feature engineering and model architecture improvements produced marginal or negative results at smaller scale, but improvements are now succeeding with expanded data.

## Impact

- Limits effectiveness of complex feature engineering (small groups, noisy statistics)
- Prevents reliable per-artist, per-venue, and per-segment modeling
- Makes ablation testing results unstable — small dataset means high variance in metrics
- Bayesian smoothing factors must be set high to compensate, pulling estimates toward global means

## Evidence

Multiple improvement attempts failed due to insufficient data:
- Section-level target encoding: only 2-3 listings per section (Issue #005)
- Segment-aware outlier capping: unstable per-segment statistics (Issue #007)
- Per-artist modeling: only 3-4 events per artist on average

## Current Mitigation

- Automated data collection pipeline running on EC2 (hourly via systemd timer)
- Collecting from VividSeats for concerts across major US cities
- Popularity data aggregated from YouTube Music and Last.fm
- Bayesian smoothing compensates for small group sizes

## Path Forward

- Continue automated collection to grow the dataset
- Prioritize artist diversity (currently 42 artists, up from 23)
- Expand geographic coverage for regional features
- Re-evaluate complex features as dataset grows past key thresholds

## Outcome

- Established data collection as the highest-ROI activity for model improvement
- Deployed EC2-based automated collection infrastructure
- Set expectation that model architecture changes have diminishing returns at current scale
- Dataset has grown from 81 events/23 artists to 136 events/42 artists as of v30, enabling previously-failed features (section-level encoding) to now contribute positively
