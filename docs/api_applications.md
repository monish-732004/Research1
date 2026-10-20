# API Access Applications

## Electricity Maps — Academic Free-Tier API

**Target:** https://www.electricitymaps.com/free-tier-api  
**Subject:** Academic research — GPU cluster carbon analysis  
**Status:** Draft (not yet submitted — review before sending)

---

### Application Text

> **Subject:** Academic research access — GPU cluster carbon analysis
>
> Dear Electricity Maps Team,
>
> I am a researcher at SRM Institute of Science and Technology working on a
> paper titled **"Quantifying the Carbon Cost of GPU Resource Fragmentation in
> Large-Scale Cloud Clusters"**, targeting HotCarbon 2027 / IEEE IGSC 2027.
>
> **Research summary:**
> We analyse the Alibaba cluster-trace-gpu-v2026 dataset (38,048 GPU servers,
> 185 days) to measure how much GPU capacity is wasted due to scheduler
> fragmentation — GPUs provisioned but never allocated to any job. We then
> quantify the associated carbon cost and evaluate whether carbon-aware
> deferral of batch workloads (the CADS algorithm) could reduce emissions.
>
> **Why we need Electricity Maps:**
> The Alibaba cluster is located in mainland China. Our current analysis uses
> synthetic diurnal carbon intensity profiles calibrated to IEA 2023 regional
> averages. To produce publication-quality results we need real historical
> hourly carbon intensity data for Chinese grid regions:
>
> - `CN-HB` (North China / Hebei — proxy for cn-north)
> - `CN-ZJ` (East China / Zhejiang — proxy for cn-east)
> - `CN-GD` (South China / Guangdong — proxy for cn-south)
> - `CN-NM` (Northwest China / Inner Mongolia — proxy for cn-northwest)
>
> **Intended use:**
> - Query historical hourly carbon intensity for the period January–July 2025
>   (covering all three calendar anchor scenarios for our 185-day trace)
> - Approximately 4 × 185 × 24 ≈ **17,760 hourly data points total**
> - No real-time or commercial use; results will be published open-access
>
> **Expected query volume:** ~50,000 historical hourly queries total
> (4 regions × 3 anchors × 4,440 hours + safety margin)
>
> **Output:** The real carbon profiles will replace our synthetic estimates and
> will be archived alongside the paper on GitHub for reproducibility.
>
> **Affiliation:** SRM Institute of Science and Technology, Chennai, India
>
> Thank you for considering this request. I am happy to provide a letter from
> my supervisor or any other documentation you require.
>
> Best regards,  
> [Your name]  
> [Your email]  
> SRM Institute of Science and Technology

---

### Notes for reviewer

- The free academic tier allows up to 50K queries/month — this request is well
  within that limit.
- If zone codes differ from the ones listed above, update `GRID_PROFILES` in
  `src/config.py` and re-run `python -m src.analysis.carbon_fetch --force`.
- When the API key is received, set `ELECTRICITY_MAPS_API_KEY` in `.env` and
  implement the `fetch_electricity_maps()` stub in `src/analysis/carbon_fetch.py`.
