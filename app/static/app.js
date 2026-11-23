// Pre-fill the five inputs with the typical values of the chosen state.
(function () {
  const prefill = JSON.parse(document.getElementById("prefill").textContent);
  const state = document.getElementById("state");
  const storm = document.getElementById("noaa_storm_count");
  const hist = document.getElementById("hist_state_storm_cust");
  const fields = ["hist_state_storm_cust", "nws_watch_active", "noaa_n_zones", "nws_hours_since_warning", "noaa_storm_count"];
  let histEdited = false;
  hist.addEventListener("input", () => { histEdited = true; });

  function values(stormFlag) {
    const s = prefill[state.value];
    return s ? s[stormFlag] || s[1 - stormFlag] : null;
  }

  function fillAll() {
    const v = values(1);                         // start from the "storm active" situation
    if (!v) return;
    fields.forEach((f) => {
      const el = document.getElementById(f);
      el.value = v[f] === null ? "" : (f === "hist_state_storm_cust" ? Math.round(v[f]) : v[f]);
    });
    histEdited = false;
  }

  // The outage history differs for outages with and without a storm: follow the storm input.
  storm.addEventListener("change", () => {
    if (histEdited || !state.value) return;
    const v = values(Number(storm.value) > 0 ? 1 : 0);
    if (v && v.hist_state_storm_cust !== null) hist.value = Math.round(v.hist_state_storm_cust);
  });

  state.addEventListener("change", fillAll);
  document.getElementById("reset").addEventListener("click", fillAll);
  if (state.value && fields.some((f) => document.getElementById(f).value === "")) fillAll();
})();
