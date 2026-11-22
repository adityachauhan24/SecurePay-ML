
import { useState, useEffect } from "react";
import Papa from "papaparse";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

import {
  LayoutDashboard,
  CreditCard,
  FileSpreadsheet,
  ShieldCheck,
  ShieldAlert,
  CheckCircle2,
  Activity,
  Database,
  BarChart3,
  Clock3,
  UploadCloud,
  Download,
  ArrowRight,
  LockKeyhole,
  BrainCircuit,
  Info,
  AlertCircle,
  TrendingUp,
  Layers3,
  RefreshCw,
} from "lucide-react";

import "./App.css";

// ==========================================
// CONFIGURATION
// ==========================================

const API_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const COLORS = {
  cyan: "#08CFE5",
  fraud: "#F04458",
  normal: "#10A77A",
  navy: "#0B1730",
};

const TRANSACTION_TYPES = [
  "TRANSFER",
  "CASH_OUT",
  "CASH_IN",
  "PAYMENT",
  "DEBIT",
];

const MENU_ITEMS = [
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Single Prediction", icon: CreditCard },
  { label: "Batch Analysis", icon: FileSpreadsheet },
];

const INITIAL_FORM = {
  step: "101",
  type: "CASH_OUT",
  amount: "250000",
  oldbalanceOrg: "250000",
  oldbalanceDest: "0",
};

const formatPercent = (value) => {
  if (typeof value !== "number") return "—";
  return `${(value * 100).toFixed(2)}%`;
};

const formatNumber = (value) => {
  if (typeof value !== "number") return "—";
  return value.toLocaleString();
};

// ==========================================
// SECUREPAY ML LOGO
// ==========================================

function SecurePayLogo({ size = 58 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      role="img"
      aria-label="SecurePay ML Logo"
    >
      <path
        d="M32 5L54 14V31C54 45 44 54 32 60C20 54 10 45 10 31V14L32 5Z"
        stroke="#08CFE5"
        strokeWidth="4.5"
        strokeLinejoin="round"
      />

      <rect
        x="23"
        y="28"
        width="18"
        height="17"
        rx="3"
        fill="#08CFE5"
      />

      <path
        d="M27 28V23a5 5 0 0110 0v5"
        stroke="#E5FBFF"
        strokeWidth="3"
        strokeLinecap="round"
      />

      <circle cx="32" cy="35" r="2" fill="#09172F" />
    </svg>
  );
}

// ==========================================
// REUSABLE UI COMPONENTS
// ==========================================

function SectionHeading({ icon: Icon, title, subtitle }) {
  return (
    <div className="section-heading">
      <div className="section-heading-icon">
        <Icon size={20} />
      </div>

      <div>
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, note, tone }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${tone}`}>
        <Icon size={24} />
      </div>

      <div>
        <p>{label}</p>
        <h2>{value}</h2>
        <span>{note}</span>
      </div>
    </div>
  );
}

function MetricCard({ label, value, explanation }) {
  return (
    <div className="metric-card">
      <p>{label}</p>
      <h3>{value}</h3>
      <span>{explanation}</span>
    </div>
  );
}

function EmptyChart() {
  return (
    <div className="empty-chart">
      <BarChart3 size={32} />

      <strong>No prediction data yet</strong>

      <p>
        Analyze transactions to populate this chart.
      </p>
    </div>
  );
}

// ==========================================
// MAIN REACT APPLICATION
// ==========================================

function App() {

  // =====================================
  // REACT STATES
  // =====================================

  const [activePage, setActivePage] = useState("Dashboard");

  const [form, setForm] = useState(INITIAL_FORM);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [csvFile, setCsvFile] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState("");
  const [batchSuccess, setBatchSuccess] = useState("");

  const [history, setHistory] = useState([]);

  const [modelInfo, setModelInfo] = useState(null);
  const [modelError, setModelError] = useState("");
  const [modelLoading, setModelLoading] = useState(true);

  // =====================================
  // FETCH MODEL PERFORMANCE FROM BACKEND
  // =====================================


useEffect(() => {
  let cancelled = false;
  let retryTimer;

  const MAX_RETRIES = 15;
  const RETRY_DELAY = 10000; // 10 seconds

  async function fetchModelInfo(attempt = 1) {
    try {
      const response = await fetch(`${API_URL}/model-info`);

      if (!response.ok) {
        throw new Error("Model information unavailable.");
      }

      const data = await response.json();

      if (!cancelled) {
        setModelInfo(data);
        setModelError("");
        setModelLoading(false);
      }

    } catch (err) {
      if (cancelled) return;

      if (attempt < MAX_RETRIES) {
        setModelLoading(true);
        setModelError("");

        retryTimer = setTimeout(() => {
          if (!cancelled) {
            fetchModelInfo(attempt + 1);
          }
        }, RETRY_DELAY);

      } else {
        setModelLoading(false);
        setModelError(
          "Unable to connect to the backend. Please refresh."
        );
      }
    }
  }

  fetchModelInfo();

  return () => {
    cancelled = true;
    clearTimeout(retryTimer);
  };
}, []);


  // =====================================
  // COMBINED TRANSACTION HISTORY
  // =====================================

  const allHistory = [...history].sort(
    (a, b) =>
      b.createdAt - a.createdAt ||
      b.order - a.order
  );

  // =====================================
  // DASHBOARD STATISTICS
  // =====================================

  const totalTransactions = allHistory.length;

  const fraudDetected = allHistory.filter(
    (transaction) => transaction.prediction === 1
  ).length;

  const normalTransactions =
    totalTransactions - fraudDetected;

  const fraudRate =
    totalTransactions > 0
      ? (fraudDetected / totalTransactions) * 100
      : 0;

  // =====================================
  // CHART 1: FRAUD DISTRIBUTION
  // =====================================

  const fraudDistribution = [
    {
      name: "Flagged Fraud",
      value: fraudDetected,
    },
    {
      name: "Predicted Normal",
      value: normalTransactions,
    },
  ];

  // =====================================
  // CHART 2: TRANSACTION TYPE ANALYSIS
  // =====================================

  const transactionTypeData = TRANSACTION_TYPES.map(
    (type) => {
      const transactions = allHistory.filter(
        (item) => item.type === type
      );

      return {
        type,

        Fraud: transactions.filter(
          (item) => item.prediction === 1
        ).length,

        Normal: transactions.filter(
          (item) => item.prediction === 0
        ).length,
      };
    }
  );

  // =====================================
  // HANDLE FORM CHANGES
  // =====================================

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    setResult(null);
    setError("");
  }

  // =====================================
  // SINGLE TRANSACTION PREDICTION
  // =====================================

  async function handlePredict(event) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    const transaction = {
      step: Number(form.step),
      type: form.type,
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      oldbalanceDest: Number(form.oldbalanceDest),
    };

    try {
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(transaction),
        }
      );

      if (!response.ok) {
        const details = await response.json().catch(
          () => null
        );

        throw new Error(
          typeof details?.detail === "string"
            ? details.detail
            : `Prediction failed (HTTP ${response.status})`
        );
      }

      const data = await response.json();

      setResult(data);

      const record = {
        id: crypto.randomUUID(),
        createdAt: Date.now(),
        order: 0,
        source: "Single",
        type: transaction.type,
        amount: transaction.amount,
        fraud_score: Number(data.fraud_score),
        prediction: Number(data.prediction),
        status: data.status,
      };

      setHistory((previous) => [
        record,
        ...previous,
      ]);

    } catch (err) {
      setError(
        err.message || "Unable to analyze transaction."
      );
    } finally {
      setLoading(false);
    }
  }

  // =====================================
  // BATCH CSV PREDICTION
  // =====================================

  async function handleBatchUpload(event) {
    event.preventDefault();

    setBatchError("");
    setBatchSuccess("");

    if (!csvFile) {
      setBatchError("Please select a CSV file.");
      return;
    }

    if (csvFile.size > 10 * 1024 * 1024) {
      setBatchError(
        "CSV file must be smaller than 10 MB."
      );
      return;
    }

    setBatchLoading(true);

    const formData = new FormData();
    formData.append("file", csvFile);

    try {
      const response = await fetch(
        `${API_URL}/predict-batch`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const details = await response.json().catch(
          () => null
        );

        throw new Error(
          typeof details?.detail === "string"
            ? details.detail
            : `Batch analysis failed (HTTP ${response.status})`
        );
      }

      const blob = await response.blob();
      const csvText = await blob.text();

      // Parse result CSV
      const parsed = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
      });

      if (parsed.errors.length > 0) {
        throw new Error(
          "Unable to read prediction CSV."
        );
      }

      const requiredColumns = [
        "type",
        "amount",
        "fraud_score",
        "prediction",
        "status",
      ];

      const missingColumns = requiredColumns.filter(
        (column) =>
          !parsed.meta.fields?.includes(column)
      );

      if (missingColumns.length > 0) {
        throw new Error(
          `Missing columns: ${missingColumns.join(", ")}`
        );
      }

      if (parsed.data.length === 0) {
        throw new Error("No predictions were returned.");
      }

      const batchTime = Date.now();

      const records = parsed.data.map((row, index) => {
        const prediction = Number(row.prediction);
        const amount = Number(row.amount);
        const fraudScore = Number(row.fraud_score);

        if (
          ![0, 1].includes(prediction) ||
          !Number.isFinite(amount) ||
          !Number.isFinite(fraudScore) ||
          fraudScore < 0 ||
          fraudScore > 1 ||
          !TRANSACTION_TYPES.includes(row.type)
        ) {
          throw new Error(
            `Invalid prediction at CSV row ${index + 2}`
          );
        }

        return {
          id: crypto.randomUUID(),
          createdAt: batchTime,
          order: index,
          source: "Batch",
          type: row.type,
          amount,
          fraud_score: fraudScore,
          prediction,
          status: prediction === 1 ? "FRAUD" : "NORMAL",
        };
      });

      // Add all batch predictions to history
      setHistory((previous) => [
        ...records,
        ...previous,
      ]);

      // Download result CSV
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "securepay_predictions.csv";

      document.body.appendChild(link);
      link.click();
      link.remove();

      setTimeout(() => {
        URL.revokeObjectURL(url);
      }, 1000);

      setBatchSuccess(
        `${records.length} transactions analyzed. CSV download initiated.`
      );

    } catch (err) {
      setBatchError(
        err.message || "Batch analysis failed."
      );
    } finally {
      setBatchLoading(false);
    }
  }

  // =====================================
  // USER INTERFACE
  // =====================================

  return (
    <div className="app-container">

      {/* =====================================
          SIDEBAR
      ===================================== */}

      <aside className="sidebar">

        <div className="sidebar-brand">
          <SecurePayLogo size={60} />

          <div>
            <h2 className="logo">
              SecurePay <span>ML</span>
            </h2>

            <p className="sidebar-subtitle">
              Intelligent Fraud Detection
            </p>
          </div>
        </div>

        <p className="menu-label">WORKSPACE</p>

        <nav className="sidebar-menu">

          {MENU_ITEMS.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.label}
                className={
                  activePage === item.label
                    ? "active"
                    : ""
                }
                onClick={() =>
                  setActivePage(item.label)
                }
              >
                <Icon size={20} />
                <span>{item.label}</span>
              </button>
            );
          })}

        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-security-icon">
            <ShieldCheck size={30} />
          </div>

          <strong>SecurePay ML</strong>

          <p>
            Powered by Core Machine Learning
          </p>
        </div>

      </aside>

      {/* =====================================
          MAIN CONTENT
      ===================================== */}

      <main className="main-content">

        {/* HEADER */}
        <header className="top-header">

          <div>
            <p className="eyebrow">
              SECUREPAY / WORKSPACE
            </p>

            <h1>{activePage}</h1>

            <p className="header-subtitle">
              Machine Learning Powered Fraud Detection
            </p>
          </div>

          
<div className="header-status">
  <span
    className={
      modelLoading
        ? "loading-dot"
        : modelInfo
          ? "online-dot"
          : "offline-dot"
    }
  />

  {modelLoading
    ? "Starting ML Backend..."
    : modelInfo
      ? "Model Info Available"
      : "Model Info Unavailable"}
</div>


        </header>

        {/* =====================================
            DASHBOARD PAGE
        ===================================== */}

        {activePage === "Dashboard" && (
          <>

            {/* HERO SECTION */}
            <section className="welcome-section">

              <div className="welcome-content">

                <span className="hero-badge">
                  <LockKeyhole size={14} />
                  SECUREPAY INTELLIGENCE
                </span>

                <h2>
                  Smarter Fraud Detection.
                  <br />
                  Safer Transactions.
                </h2>

                <p>
                  Explore ML model performance,
                  analyze transaction behavior,
                  and identify suspicious activity
                  using our trained XGBoost classifier.
                </p>

                <button
                  className="hero-button"
                  onClick={() =>
                    setActivePage("Single Prediction")
                  }
                >
                  Analyze Transaction
                  <ArrowRight size={18} />
                </button>

              </div>

              <div className="hero-icon">
                <ShieldCheck
                  size={115}
                  strokeWidth={1.2}
                />
              </div>

            </section>

            {/* =====================================
                DASHBOARD STATISTICS
            ===================================== */}

            <section className="stats-grid">

              <StatCard
                icon={Database}
                label="Total Analyzed"
                value={formatNumber(totalTransactions)}
                note="Current browser session"
                tone="cyan"
              />

              <StatCard
                icon={ShieldAlert}
                label="Flagged Fraud"
                value={formatNumber(fraudDetected)}
                note="Predicted suspicious"
                tone="red"
              />

              <StatCard
                icon={CheckCircle2}
                label="Predicted Normal"
                value={formatNumber(normalTransactions)}
                note="Classified as normal"
                tone="green"
              />

            </section>

            {/* =====================================
                TWO OPERATIONAL CHARTS
            ===================================== */}

            <section className="chart-grid">

              {/* FRAUD DISTRIBUTION */}
              <div className="panel">

                <SectionHeading
                  icon={Activity}
                  title="Fraud Distribution"
                  subtitle="Fraud vs normal predictions"
                />

                {totalTransactions === 0 ? (
                  <EmptyChart />
                ) : (
                  <div className="donut-container">

                    <ResponsiveContainer
                      width="100%"
                      height={280}
                    >
                      <PieChart>

                        <Pie
                          data={fraudDistribution}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="46%"
                          innerRadius={72}
                          outerRadius={100}
                          paddingAngle={2}
                          stroke="none"
                        >
                          <Cell fill={COLORS.fraud} />
                          <Cell fill={COLORS.normal} />
                        </Pie>

                        <Tooltip />
                        <Legend />

                      </PieChart>
                    </ResponsiveContainer>

                    <div className="donut-label">
                      <strong>
                        {fraudRate.toFixed(1)}%
                      </strong>

                      <span>Flagged</span>
                    </div>

                  </div>
                )}

              </div>

              {/* TRANSACTION TYPES */}
              <div className="panel">

                <SectionHeading
                  icon={BarChart3}
                  title="Transaction Type Analysis"
                  subtitle="Predictions grouped by transaction type"
                />

                {totalTransactions === 0 ? (
                  <EmptyChart />
                ) : (
                  <div className="chart-container">

                    <ResponsiveContainer
                      width="100%"
                      height={300}
                    >
                      <BarChart
                        data={transactionTypeData}
                        margin={{
                          top: 10,
                          right: 10,
                          left: -25,
                          bottom: 30,
                        }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          vertical={false}
                          stroke="#E6EDF3"
                        />

                        <XAxis
                          dataKey="type"
                          angle={-20}
                          textAnchor="end"
                          height={65}
                          tick={{ fontSize: 10 }}
                        />

                        <YAxis
                          allowDecimals={false}
                          tick={{ fontSize: 11 }}
                        />

                        <Tooltip />
                        <Legend />

                        <Bar
                          dataKey="Fraud"
                          fill={COLORS.fraud}
                          radius={[5, 5, 0, 0]}
                          maxBarSize={28}
                        />

                        <Bar
                          dataKey="Normal"
                          fill={COLORS.normal}
                          radius={[5, 5, 0, 0]}
                          maxBarSize={28}
                        />

                      </BarChart>
                    </ResponsiveContainer>

                  </div>
                )}

              </div>

            </section>

            {/* =====================================
                MODEL PERFORMANCE
            ===================================== */}

            <section className="panel model-performance">

              <div className="model-section-header">

                <SectionHeading
                  icon={BrainCircuit}
                  title="Model Performance"
                  subtitle="Historical offline test evaluation"
                />

                <span className="model-chip">
                  XGBoost Classifier
                </span>

              </div>

              {modelLoading && (
  <p>Starting ML Backend... This may take a minute.</p>
)}

              {modelError && (
                <div className="error-message">
                  <AlertCircle size={18} />
                  {modelError}
                </div>
              )}

              {modelInfo && (
                <>

                  <div className="model-metrics-grid">

                    <MetricCard
                      label="Test Accuracy"
                      value={formatPercent(
                        modelInfo.test_metrics.accuracy
                      )}
                      explanation="Overall correct classifications"
                    />

                    <MetricCard
                      label="Precision"
                      value={formatPercent(
                        modelInfo.test_metrics.precision
                      )}
                      explanation="Accuracy of fraud alerts"
                    />

                    <MetricCard
                      label="Recall"
                      value={formatPercent(
                        modelInfo.test_metrics.recall
                      )}
                      explanation="Actual frauds detected"
                    />

                    <MetricCard
                      label="F1-Score"
                      value={formatPercent(
                        modelInfo.test_metrics.f1_score
                      )}
                      explanation="Precision–recall balance"
                    />

                    <MetricCard
                      label="F2-Score"
                      value={formatPercent(
                        modelInfo.test_metrics.f2_score
                      )}
                      explanation="Recall-weighted performance"
                    />

                    <MetricCard
                      label="Average Precision"
                      value={formatPercent(
                        modelInfo.test_metrics.average_precision
                      )}
                      explanation="Precision–recall ranking quality"
                    />

                  </div>

                  {/* YOUR REQUESTED ACCURACY NOTE */}
                  <div className="accuracy-explanation">

                    <Info size={22} />

                    <div>

                      <h4>
                        Why Accuracy Alone Can Be Misleading
                      </h4>

                      <p>
                        Our XGBoost model achieved
                        <strong>
                          {" "}
                          {formatPercent(
                            modelInfo.test_metrics.accuracy
                          )}
                          {" "}test accuracy
                        </strong>.
                        However, the PaySim dataset
                        is highly imbalanced, with
                        approximately 99.87% normal
                        transactions.
                      </p>

                      <p>
                        Even a model predicting every
                        transaction as NORMAL would
                        achieve approximately 99.87%
                        accuracy.
                      </p>

                      <p>
                        Therefore,
                        <strong>
                          {" "}Precision, Recall,
                          F1-Score, and Average Precision
                        </strong>
                        {" "}provide more meaningful
                        insights into fraud detection
                        performance.
                      </p>

                      <small>
                        {modelInfo.evaluation_note}
                      </small>

                    </div>

                  </div>

                </>
              )}

            </section>

            {/* =====================================
                MODEL COMPARISON + ABOUT MODEL
            ===================================== */}

            <section className="model-details-grid">

              {/* MODEL COMPARISON */}
              <div className="panel">

                <SectionHeading
                  icon={TrendingUp}
                  title="Model Comparison"
                  subtitle="Best observed validation F1-Score"
                />

                {modelInfo?.model_comparison ? (
                  <div className="comparison-list">

                    {modelInfo.model_comparison.map(
                      (item) => (
                        <div
                          className="comparison-item"
                          key={item.model}
                        >

                          <div className="comparison-label">
                            <strong>{item.model}</strong>

                            <span>
                              {formatPercent(
                                item.validation_f1
                              )}
                            </span>
                          </div>

                          <div className="progress-track">

                            <div
                              className={
                                item.model === "XGBoost"
                                  ? "progress-fill selected"
                                  : "progress-fill"
                              }
                              style={{
                                width: `${
                                  item.validation_f1 * 100
                                }%`
                              }}
                            />

                          </div>

                        </div>
                      )
                    )}

                    <p className="comparison-note">
                      Comparison uses validation
                      results from model experiments.
                      These are not independent
                      test-set results.
                    </p>

                  </div>
                ) : (
                  <p className="empty-text">
                    Model comparison unavailable.
                  </p>
                )}

              </div>

              {/* ABOUT MODEL */}
              <div className="panel">

                <SectionHeading
                  icon={Layers3}
                  title="About the Model"
                  subtitle="Model architecture and dataset"
                />

                {modelInfo ? (
                  <div className="about-grid">

                    <div>
                      <span>Algorithm</span>
                      <strong>
                        {modelInfo.algorithm}
                      </strong>
                    </div>

                    <div>
                      <span>ML Approach</span>
                      <strong>
                        {modelInfo.approach}
                      </strong>
                    </div>

                    <div>
                      <span>Training Sample</span>
                      <strong>
                        {formatNumber(
                          modelInfo.training_samples
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Input Features</span>
                      <strong>
                        {modelInfo.feature_count}
                      </strong>
                    </div>

                    <div>
                      <span>Decision Threshold</span>
                      <strong>
                        {modelInfo.threshold}
                      </strong>
                    </div>

                    <div>
                      <span>Dataset</span>
                      <strong>
                        PaySim Synthetic
                      </strong>
                    </div>

                  </div>
                ) : (
                  <p className="empty-text">
                    Model information unavailable.
                  </p>
                )}

              </div>

            </section>

            {/* =====================================
                RECENT PREDICTIONS
            ===================================== */}

            <section className="panel history-section">

              <SectionHeading
                icon={Clock3}
                title="Recent Predictions"
                subtitle="Latest analyzed transactions"
              />

              {allHistory.length === 0 ? (
                <p className="empty-text">
                  No transactions analyzed yet.
                  Use Single Prediction or Batch Analysis
                  to generate predictions.
                </p>
              ) : (
                <div className="table-wrapper">

                  <table className="history-table">

                    <thead>
                      <tr>
                        <th>Source</th>
                        <th>Type</th>
                        <th>Amount</th>
                        <th>Fraud Score</th>
                        <th>Prediction</th>
                      </tr>
                    </thead>

                    <tbody>

                      {allHistory.slice(0, 10).map(
                        (item) => (
                          <tr key={item.id}>

                            <td>{item.source}</td>

                            <td>{item.type}</td>

                            <td>
                              {formatNumber(item.amount)}
                            </td>

                            <td>
                              {item.fraud_score.toFixed(6)}
                            </td>

                            <td>
                              <span
                                className={
                                  item.prediction === 1
                                    ? "status-badge fraud"
                                    : "status-badge normal"
                                }
                              >
                                {item.status}
                              </span>
                            </td>

                          </tr>
                        )
                      )}

                    </tbody>

                  </table>

                </div>
              )}

            </section>

            <p className="dashboard-disclaimer">
              <Info size={16} />
              Dashboard counts represent model
              predictions from this browser session,
              not confirmed real-world fraud.
            </p>

          </>
        )}

        {/* =====================================
            SINGLE PREDICTION PAGE
        ===================================== */}

        {activePage === "Single Prediction" && (
          <section className="panel form-page">

            <SectionHeading
              icon={CreditCard}
              title="Transaction Risk Assessment"
              subtitle="Enter details to generate an ML fraud prediction"
            />

            <form
              className="prediction-form"
              onSubmit={handlePredict}
            >

              <div className="form-group">
                <label htmlFor="step">
                  Transaction Step
                </label>

                <input
                  id="step"
                  name="step"
                  type="number"
                  min="1"
                  step="1"
                  required
                  value={form.step}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label htmlFor="type">
                  Transaction Type
                </label>

                <select
                  id="type"
                  name="type"
                  value={form.type}
                  onChange={handleChange}
                >
                  {TRANSACTION_TYPES.map(
                    (type) => (
                      <option
                        value={type}
                        key={type}
                      >
                        {type}
                      </option>
                    )
                  )}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="amount">
                  Transaction Amount
                </label>

                <input
                  id="amount"
                  name="amount"
                  type="number"
                  min="0"
                  step="any"
                  required
                  value={form.amount}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label htmlFor="oldbalanceOrg">
                  Sender Old Balance
                </label>

                <input
                  id="oldbalanceOrg"
                  name="oldbalanceOrg"
                  type="number"
                  min="0"
                  step="any"
                  required
                  value={form.oldbalanceOrg}
                  onChange={handleChange}
                />
              </div>

              <div className="form-group">
                <label htmlFor="oldbalanceDest">
                  Receiver Old Balance
                </label>

                <input
                  id="oldbalanceDest"
                  name="oldbalanceDest"
                  type="number"
                  min="0"
                  step="any"
                  required
                  value={form.oldbalanceDest}
                  onChange={handleChange}
                />
              </div>

              <button
                type="submit"
                className="primary-button"
                disabled={loading}
              >
                {loading
                  ? "Analyzing..."
                  : "Analyze Transaction"}

                <ArrowRight size={18} />
              </button>

            </form>

            {error && (
              <div className="error-message" role="alert">
                <AlertCircle size={18} />
                {error}
              </div>
            )}

            {result && (
              <div
                className={
                  result.prediction === 1
                    ? "prediction-result fraud"
                    : "prediction-result normal"
                }
              >

                <div className="result-title">

                  {result.prediction === 1
                    ? <ShieldAlert size={28} />
                    : <ShieldCheck size={28} />
                  }

                  <h3>
                    {result.prediction === 1
                      ? "Suspicious Transaction Detected"
                      : "Transaction Classified as Normal"
                    }
                  </h3>

                </div>

                <div className="result-row">
                  <span>Fraud Risk Score</span>

                  <strong>
                    {Number(
                      result.fraud_score
                    ).toFixed(6)}
                  </strong>
                </div>

                <div className="result-row">
                  <span>Prediction</span>

                  <strong>{result.status}</strong>
                </div>

                <div className="result-row">
                  <span>Decision Threshold</span>

                  <strong>
                    {result.threshold}
                  </strong>
                </div>

                <p className="risk-note">
                  Risk scores are not calibrated
                  real-world fraud probabilities.
                </p>

              </div>
            )}

          </section>
        )}

        {/* =====================================
            BATCH ANALYSIS PAGE
        ===================================== */}

        {activePage === "Batch Analysis" && (
          <section className="panel form-page">

            <SectionHeading
              icon={FileSpreadsheet}
              title="Batch Transaction Analysis"
              subtitle="Upload a CSV to predict multiple transactions"
            />

            <form
              className="batch-form"
              onSubmit={handleBatchUpload}
            >

              <label
                htmlFor="csv-upload"
                className="upload-zone"
              >

                <UploadCloud size={43} />

                <strong>
                  {csvFile
                    ? csvFile.name
                    : "Choose a CSV File"}
                </strong>

                <span>
                  CSV format only · Maximum 10 MB
                </span>

              </label>

              <input
                id="csv-upload"
                className="hidden-file"
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => {
                  setCsvFile(
                    event.target.files?.[0] || null
                  );

                  setBatchError("");
                  setBatchSuccess("");
                }}
              />

              <button
                className="primary-button"
                type="submit"
                disabled={batchLoading || !csvFile}
              >
                {batchLoading
                  ? "Analyzing CSV..."
                  : "Analyze CSV & Download Results"
                }

                <Download size={18} />
              </button>

            </form>

            {batchError && (
              <div className="error-message" role="alert">
                <AlertCircle size={18} />
                {batchError}
              </div>
            )}

            {batchSuccess && (
              <div className="success-message" role="status">
                <CheckCircle2 size={18} />
                {batchSuccess}
              </div>
            )}

            <div className="csv-info">

              <h3>Required CSV Columns</h3>

              <code>
                step,type,amount,oldbalanceOrg,oldbalanceDest
              </code>

              <p>
                Output includes fraud_score,
                prediction, and status columns.
              </p>

            </div>

          </section>
        )}

        {/* FOOTER */}
        <footer className="app-footer">
          <span>
            SecurePay ML · Synthetic Transaction
            Fraud Detection
          </span>

          <span>
            React + FastAPI + XGBoost
          </span>
        </footer>

      </main>

    </div>
  );
}

export default App;
