import React, { useState, useEffect } from "react";
import axios from "axios";

const API_URL =
  process.env.REACT_APP_BACKEND_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`;

const defaultForm = {
  Country: "USA",
  Year: 2022,
  Target_Industry: "Finance",
  Financial_Loss: 50.0,
  Number_of_Affected_Users: 100000,
  Attack_Source: "Nation-state",
  Security_Vulnerability_Type: "Unpatched Software",
  Defense_Mechanism_Used: "Firewall",
  Incident_Resolution_Time: 48,
};

export default function Home() {
  const [form, setForm]       = useState(defaultForm);
  const [options, setOptions] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    axios.get(`${API_URL}/options`)
      .then(res => setOptions(res.data))
      .catch(() => setError("Cannot connect to backend."));
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    const integerFields = ["Year", "Number_of_Affected_Users", "Incident_Resolution_Time"];
    const floatFields = ["Financial_Loss"];

    let parsedValue = value;
    if (integerFields.includes(name)) {
      parsedValue = value === "" ? "" : parseInt(value, 10);
    } else if (floatFields.includes(name)) {
      parsedValue = value === "" ? "" : parseFloat(value);
    }

    setForm(prev => ({ ...prev, [name]: parsedValue }));
  };

  const handleSubmit = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, form);
      setResult(res.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail ? `Prediction failed: ${detail}` : "Prediction failed. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  const labelMap = {
    Country: "Country",
    Target_Industry: "Target Industry",
    Attack_Source: "Attack Source",
    Security_Vulnerability_Type: "Security Vulnerability Type",
    Defense_Mechanism_Used: "Defense Mechanism Used",
  };

  const isAttack = result?.verdict === "Attack";

  return (
    <div style={{ fontFamily:"Arial,sans-serif", background:"#f0f2f5", minHeight:"100vh" }}>
      <header style={{ background:"#1a1a2e", color:"white", padding:"20px 40px" }}>
        <h1 style={{ margin:0, fontSize:22 }}>🔒 Cybersecurity Attack Detector</h1>
        <p style={{ margin:"4px 0 0", color:"#aab", fontSize:13 }}>
          MOD10 Machine Learning · Binary Classification · Attack vs. Normal
        </p>
      </header>

      <div style={{ maxWidth:920, margin:"28px auto", padding:"0 20px" }}>
        {error && <div style={{ background:"#fee", border:"1px solid #e88", borderRadius:8, padding:"10px 16px", marginBottom:18, color:"#c00" }}>⚠️ {error}</div>}

        {/* Form */}
        <div style={{ background:"white", borderRadius:12, padding:26, boxShadow:"0 2px 12px rgba(0,0,0,0.08)", marginBottom:22 }}>
          <h2 style={{ marginTop:0, color:"#1a1a2e" }}>Enter Event Details</h2>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"14px 22px" }}>
            {Object.entries(labelMap).map(([field, label]) => (
              <div key={field}>
                <label style={{ display:"block", marginBottom:4, fontWeight:"bold", fontSize:13, color:"#444" }}>{label}</label>
                <select name={field} value={form[field]} onChange={handleChange}
                  style={{ width:"100%", padding:"8px 10px", borderRadius:6, border:"1px solid #ccc", fontSize:14 }}>
                  {options && options[label]
                    ? options[label].map(o => <option key={o}>{o}</option>)
                    : <option>{form[field]}</option>}
                </select>
              </div>
            ))}
            {[
              { name:"Year", label:"Year", min:2015, max:2024 },
              { name:"Financial_Loss", label:"Financial Loss (Million $)", min:0 },
              { name:"Number_of_Affected_Users", label:"Affected Users", min:0 },
              { name:"Incident_Resolution_Time", label:"Resolution Time (Hours)", min:0 },
            ].map(({ name, label, min, max }) => (
              <div key={name}>
                <label style={{ display:"block", marginBottom:4, fontWeight:"bold", fontSize:13, color:"#444" }}>{label}</label>
                <input type="number" name={name} value={form[name]} min={min} max={max}
                  onChange={handleChange}
                  style={{ width:"100%", padding:"8px 10px", borderRadius:6, border:"1px solid #ccc", fontSize:14, boxSizing:"border-box" }}/>
              </div>
            ))}
          </div>
          <button onClick={handleSubmit} disabled={loading}
            style={{ marginTop:22, padding:"11px 30px", fontSize:15, fontWeight:"bold",
              background:loading?"#aaa":"#1a1a2e", color:"white", border:"none", borderRadius:8, cursor:loading?"not-allowed":"pointer" }}>
            {loading ? "Analysing…" : "🔍 Detect Attack"}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div style={{ background:"white", borderRadius:12, padding:26, boxShadow:"0 2px 12px rgba(0,0,0,0.08)" }}>
            <h2 style={{ marginTop:0, color:"#1a1a2e" }}>Detection Result</h2>

            {/* Verdict banner */}
            <div style={{
              textAlign:"center", padding:"18px", borderRadius:10, marginBottom:22,
              background:isAttack?"#ffeaea":"#eafaf1",
              border:`2px solid ${isAttack?"#e74c3c":"#2ecc71"}`
            }}>
              <div style={{ fontSize:36 }}>{isAttack ? "🚨" : "✅"}</div>
              <div style={{ fontSize:26, fontWeight:"bold", color:isAttack?"#c0392b":"#27ae60" }}>
                {result.verdict}
              </div>
              <div style={{ color:"#555", marginTop:4 }}>
                Confidence: <strong>{result.confidence}%</strong> (Random Forest)
              </div>
            </div>

            {/* Model votes */}
            <h3 style={{ color:"#333", marginBottom:10 }}>Model Votes</h3>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12, marginBottom:22 }}>
              {Object.entries(result.predictions).map(([model, label]) => (
                <div key={model} style={{
                  textAlign:"center", padding:"14px 10px", borderRadius:10,
                  background:label==="Attack"?"#ffeaea":"#eafaf1",
                  border:`1px solid ${label==="Attack"?"#e74c3c":"#2ecc71"}`
                }}>
                  <div style={{ fontSize:11, color:"#888", marginBottom:4 }}>
                    {model.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase())}
                  </div>
                  <div style={{ fontSize:17, fontWeight:"bold", color:label==="Attack"?"#c0392b":"#27ae60" }}>
                    {label}
                  </div>
                </div>
              ))}
            </div>

            {/* Probability bars */}
            <h3 style={{ color:"#333", marginBottom:10 }}>Attack Probability (Random Forest)</h3>
            {Object.entries(result.probabilities).map(([cls, prob]) => (
              <div key={cls} style={{ marginBottom:10 }}>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:13, marginBottom:3 }}>
                  <span>{cls}</span><span>{(prob*100).toFixed(1)}%</span>
                </div>
                <div style={{ background:"#eee", borderRadius:4, height:14 }}>
                  <div style={{
                    width:`${prob*100}%`, height:"100%", borderRadius:4, transition:"width 0.4s",
                    background:cls==="Attack"?"#e74c3c":"#2ecc71"
                  }}/>
                </div>
              </div>
            ))}

            {/* SHAP explanation */}
            <h3 style={{ color:"#333", marginTop:20, marginBottom:10 }}>
              🔍 Top Features Driving This Prediction (SHAP)
            </h3>
            <p style={{ color:"#666", fontSize:13, marginBottom:12 }}>
              Positive values push toward <strong>Attack</strong>. Negative values push toward <strong>Normal</strong>.
            </p>
            {result.shap_top_features.map(({ feature, impact }) => {
              const pct = Math.min(Math.abs(impact) * 400, 100);
              return (
                <div key={feature} style={{ marginBottom:10 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", fontSize:13, marginBottom:3 }}>
                    <span>{feature.replace(/_enc$/,"").replace(/_/g," ")}</span>
                    <span style={{ color:impact>0?"#e74c3c":"#2ecc71", fontWeight:"bold" }}>
                      {impact>0?"+":""}{impact.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ background:"#eee", borderRadius:4, height:12 }}>
                    <div style={{
                      width:`${pct}%`, height:"100%", borderRadius:4,
                      background:impact>0?"#e74c3c":"#2ecc71"
                    }}/>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <footer style={{ textAlign:"center", padding:14, color:"#888", fontSize:12 }}>
        MOD10 Machine Learning Project · Winter 2026 · Instructor: Mohammed A. Shehab
      </footer>
    </div>
  );
}
