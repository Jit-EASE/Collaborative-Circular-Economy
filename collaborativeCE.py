import streamlit as st
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# -------------------------
# CONFIG
# -------------------------

st.set_page_config(
    page_title="Collaborative Consumption – Just CE Self-Learning Model",
    layout="wide",
)


# -------------------------
# MODEL MANAGER
# -------------------------

@dataclass
class CollaborativeCEModelManager:
    """
    Manages:
      - Feature transforms (shared for all heads)
      - Multi-head models:
          * Just-CE score
          * Adoption next period
          * Growth next period
          * Performance composite
      - Basic contextual bandit-like incentive recommendation
      - Self-audit metrics
    """

    feature_cols: List[str] = field(default_factory=list)
    incentive_cols: List[str] = field(default_factory=list)
    org_col: Optional[str] = None

    target_just_ce: Optional[str] = None
    target_adoption: Optional[str] = None
    target_growth: Optional[str] = None
    target_performance: Optional[str] = None

    # Transformers and models
    preprocessor: Optional[ColumnTransformer] = None

    model_just_ce: Optional[RandomForestRegressor] = None
    model_adoption: Optional[SGDRegressor] = None
    model_growth: Optional[SGDRegressor] = None
    model_performance: Optional[RandomForestRegressor] = None

    # Self-audit metrics
    metrics_history: List[Dict[str, float]] = field(default_factory=list)

    # Reward weights for bandit
    beta_A: float = 0.5
    beta_J: float = 0.5

    def _build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        # Determine numeric vs categorical
        numeric_features = []
        categorical_features = []

        for col in self.feature_cols + self.incentive_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        transformers = []

        if numeric_features:
            transformers.append(
                ("num", StandardScaler(), numeric_features)
            )

        if categorical_features:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return preprocessor

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit all heads on the current dataset (batch training).
        """

        # Sanity check
        required_targets = [
            self.target_just_ce,
            self.target_adoption,
            self.target_growth,
            self.target_performance,
        ]
        missing_targets = [t for t in required_targets if t and t not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns in data: {missing_targets}")

        # Build preprocessor if not present or features changed
        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor(df)

        X = df[self.feature_cols + self.incentive_cols].copy()

        # Fit preprocessor
        X_pre = self.preprocessor.fit_transform(X)

        # Prepare targets
        yJ = df[self.target_just_ce].values if self.target_just_ce else None
        yA = df[self.target_adoption].values if self.target_adoption else None
        yG = df[self.target_growth].values if self.target_growth else None
        yP = df[self.target_performance].values if self.target_performance else None

        # Models
        if yJ is not None:
            self.model_just_ce = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )
            self.model_just_ce.fit(X_pre, yJ)

        if yA is not None:
            self.model_adoption = SGDRegressor(
                max_iter=2000,
                tol=1e-3,
                penalty="l2",
                random_state=42,
            )
            self.model_adoption.fit(X_pre, yA)

        if yG is not None:
            self.model_growth = SGDRegressor(
                max_iter=2000,
                tol=1e-3,
                penalty="l2",
                random_state=42,
            )
            self.model_growth.fit(X_pre, yG)

        if yP is not None:
            self.model_performance = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )
            self.model_performance.fit(X_pre, yP)

        # Compute basic training metrics for audit
        metrics = self._compute_training_metrics(df)
        self.metrics_history.append(metrics)

    def partial_update(self, df_new: pd.DataFrame) -> None:
        """
        Simple 'self-learning' hook:
        Refit models on the enlarged dataset.
        In a more advanced setup you can do partial_fit for SGD models,
        or Bayesian updating.
        """
        # For now, simply call fit() on full data.
        self.fit(df_new)

    def _compute_training_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute RMSE on the training data for each head.
        """

        X = df[self.feature_cols + self.incentive_cols].copy()
        X_pre = self.preprocessor.transform(X)

        metrics = {}

        if self.target_just_ce and self.model_just_ce is not None:
            y = df[self.target_just_ce].values
            y_hat = self.model_just_ce.predict(X_pre)
            metrics["rmse_just_ce"] = sqrt(mean_squared_error(y, y_hat))

        if self.target_adoption and self.model_adoption is not None:
            y = df[self.target_adoption].values
            y_hat = self.model_adoption.predict(X_pre)
            metrics["rmse_adoption"] = sqrt(mean_squared_error(y, y_hat))

        if self.target_growth and self.model_growth is not None:
            y = df[self.target_growth].values
            y_hat = self.model_growth.predict(X_pre)
            metrics["rmse_growth"] = sqrt(mean_squared_error(y, y_hat))

        if self.target_performance and self.model_performance is not None:
            y = df[self.target_performance].values
            y_hat = self.model_performance.predict(X_pre)
            metrics["rmse_performance"] = sqrt(mean_squared_error(y, y_hat))

        return metrics

    def predict_all(self, df_input: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict all heads on new input scenarios.
        """

        X = df_input[self.feature_cols + self.incentive_cols].copy()
        X_pre = self.preprocessor.transform(X)

        out = {}

        if self.model_just_ce is not None:
            out["just_ce"] = self.model_just_ce.predict(X_pre)
        if self.model_adoption is not None:
            out["adoption_next"] = self.model_adoption.predict(X_pre)
        if self.model_growth is not None:
            out["growth_next"] = self.model_growth.predict(X_pre)
        if self.model_performance is not None:
            out["performance"] = self.model_performance.predict(X_pre)

        return out

    def recommend_incentives(
        self,
        base_row: pd.Series,
        incentive_options: List[str],
        exploration_epsilon: float = 0.1,
    ) -> Tuple[str, float, float, float]:
        """
        Simple contextual bandit-like policy:
        - For a given base context (features without incentives applied),
          evaluate each incentive (one-hot) as a candidate.
        - Compute predicted adoption + Just-CE.
        - Reward = beta_A * adoption + beta_J * just_ce.
        - With probability epsilon, explore random incentive.
        Returns: best_incentive_name, predicted_reward, predicted_adoption, predicted_just_ce.
        """
        if not incentive_options:
            raise ValueError("No incentive options provided.")

        if self.preprocessor is None or self.model_adoption is None or self.model_just_ce is None:
            raise ValueError("Models not trained yet.")

        # Exploration
        if np.random.rand() < exploration_epsilon:
            chosen = np.random.choice(incentive_options)
            mode = "explore"
        else:
            mode = "exploit"
            chosen = None

        rewards = {}
        preds_adopt = {}
        preds_just = {}

        for inc in incentive_options:
            row_copy = base_row.copy()
            # Set all incentives to 0
            for c in self.incentive_cols:
                if c in row_copy.index:
                    row_copy[c] = 0
            # Activate candidate incentive
            if inc in row_copy.index:
                row_copy[inc] = 1

            X_df = pd.DataFrame([row_copy])
            X_pre = self.preprocessor.transform(X_df[self.feature_cols + self.incentive_cols])

            pred_A = float(self.model_adoption.predict(X_pre)[0])
            pred_J = float(self.model_just_ce.predict(X_pre)[0])

            reward = self.beta_A * pred_A + self.beta_J * pred_J
            rewards[inc] = reward
            preds_adopt[inc] = pred_A
            preds_just[inc] = pred_J

        if mode == "exploit" or chosen is None:
            best_inc = max(rewards, key=rewards.get)
        else:
            best_inc = chosen

        return (
            best_inc,
            rewards[best_inc],
            preds_adopt[best_inc],
            preds_just[best_inc],
        )


# -------------------------
# SESSION INIT
# -------------------------

if "data" not in st.session_state:
    st.session_state["data"] = None

if "model_manager" not in st.session_state:
    st.session_state["model_manager"] = None


# -------------------------
# UI – SIDEBAR
# -------------------------

st.sidebar.title("Collaborative Consumption – Just CE Engine")

st.sidebar.markdown(
    """
This app implements a **self-learning multi-head model** for collaborative consumption
and just circular economy outcomes in Ireland (or any country, if you adapt the data).

1. Upload or load data  
2. Select features & targets  
3. Train models  
4. Use the recommender & self-audit tabs
"""
)

# -------------------------
# DATA TAB
# -------------------------

tab_data, tab_model, tab_recommend, tab_audit = st.tabs(
    ["1. Data & Config", "2. Train / Update Models", "3. Recommender", "4. Self-Audit"]
)

with tab_data:
    st.header("Step 1 – Load and Configure Data")

    uploaded = st.file_uploader("Upload CSV file of initiatives", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["data"] = df
        st.success("Data loaded.")
    elif st.session_state["data"] is not None:
        df = st.session_state["data"]
        st.info("Using previously loaded data from session.")
    else:
        df = None
        st.warning("No data loaded yet.")

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(20))

        all_cols = list(df.columns)

        st.markdown("### Select Feature Columns (X)")
        feature_cols = st.multiselect(
            "Choose columns that describe the initiative and context (excluding incentives & targets)",
            all_cols,
            default=[c for c in all_cols if "feat_" in c or "x_" in c][:10],
        )

        st.markdown("### Select Incentive Columns")
        incentive_cols = st.multiselect(
            "Choose columns that represent incentive flags / intensities",
            [c for c in all_cols if "inc_" in c or "incentive_" in c],
        )

        st.markdown("### Select Organisational Column (optional)")
        org_col = st.selectbox(
            "Organisation type column (optional)",
            ["<none>"] + all_cols,
            index=0,
        )
        if org_col == "<none>":
            org_col = None

        st.markdown("### Select Target Columns (Y)")
        col1, col2 = st.columns(2)

        with col1:
            target_just_ce = st.selectbox(
                "Just CE score target",
                ["<none>"] + all_cols,
                index=all_cols.index("just_ce_score") + 1 if "just_ce_score" in all_cols else 0,
            )
            if target_just_ce == "<none>":
                target_just_ce = None

            target_adoption = st.selectbox(
                "Adoption next-period target",
                ["<none>"] + all_cols,
                index=all_cols.index("adoption_next") + 1 if "adoption_next" in all_cols else 0,
            )
            if target_adoption == "<none>":
                target_adoption = None

        with col2:
            target_growth = st.selectbox(
                "Growth next-period target",
                ["<none>"] + all_cols,
                index=all_cols.index("growth_next") + 1 if "growth_next" in all_cols else 0,
            )
            if target_growth == "<none>":
                target_growth = None

            target_performance = st.selectbox(
                "Performance composite target",
                ["<none>"] + all_cols,
                index=all_cols.index("performance_score") + 1 if "performance_score" in all_cols else 0,
            )
            if target_performance == "<none>":
                target_performance = None

        st.markdown("### Save Configuration")
        if st.button("Save config to session", type="primary"):
            if not feature_cols:
                st.error("You must select at least one feature column.")
            else:
                mm = CollaborativeCEModelManager(
                    feature_cols=feature_cols,
                    incentive_cols=incentive_cols,
                    org_col=org_col,
                    target_just_ce=target_just_ce,
                    target_adoption=target_adoption,
                    target_growth=target_growth,
                    target_performance=target_performance,
                )
                st.session_state["model_manager"] = mm
                st.success("Configuration saved. Proceed to 'Train / Update Models' tab.")


# -------------------------
# MODEL TRAINING TAB
# -------------------------

with tab_model:
    st.header("Step 2 – Train / Update Self-Learning Models")

    df = st.session_state.get("data", None)
    mm: CollaborativeCEModelManager = st.session_state.get("model_manager", None)

    if df is None:
        st.warning("Please load data in the 'Data & Config' tab first.")
    elif mm is None:
        st.warning("Please define configuration in the 'Data & Config' tab first.")
    else:
        st.subheader("Current Configuration")
        st.json(
            {
                "feature_cols": mm.feature_cols,
                "incentive_cols": mm.incentive_cols,
                "org_col": mm.org_col,
                "target_just_ce": mm.target_just_ce,
                "target_adoption": mm.target_adoption,
                "target_growth": mm.target_growth,
                "target_performance": mm.target_performance,
            }
        )

        train_button_col, update_button_col = st.columns(2)

        with train_button_col:
            if st.button("Initial Train / Refit on Full Dataset", type="primary"):
                try:
                    mm.fit(df)
                    st.session_state["model_manager"] = mm
                    st.success("Models trained on full dataset.")
                except Exception as e:
                    st.error(f"Error during training: {e}")

        with update_button_col:
            st.markdown(
                """
If you append new rows to your dataset and re-upload,
you can call **Partial Update** to simulate self-learning.
Internally this refits on the current dataset; you can
switch to partial_fit if you want stricter online learning.
"""
            )
            if st.button("Partial Update (Self-Learning Step)"):
                try:
                    mm.partial_update(df)
                    st.session_state["model_manager"] = mm
                    st.success("Models updated (self-learning step).")
                except Exception as e:
                    st.error(f"Error during partial update: {e}")

        if mm.metrics_history:
            st.markdown("### Latest Training Metrics")
            st.json(mm.metrics_history[-1])


# -------------------------
# RECOMMENDER TAB
# -------------------------

with tab_recommend:
    st.header("Step 3 – Incentive & Outcome Recommender")

    df = st.session_state.get("data", None)
    mm: CollaborativeCEModelManager = st.session_state.get("model_manager", None)

    if df is None or mm is None or mm.preprocessor is None:
        st.warning("You need data, configuration and trained models first.")
    else:
        st.markdown(
            """
Provide a **hypothetical initiative context** (features and optional base incentives).
The engine will:
- Predict Just-CE, adoption, growth and performance.
- Recommend the best single incentive (contextual bandit-style) out of selected options.
"""
        )

        # Build input form based on feature & incentive columns
        default_row = df.iloc[0].copy()

        with st.form("scenario_form"):
            st.subheader("Scenario Inputs")

            scenario_values = {}

            for col in mm.feature_cols + mm.incentive_cols:
                if col not in df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    scenario_values[col] = st.slider(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                    )
                else:
                    unique_vals = df[col].dropna().unique().tolist()
                    if not unique_vals:
                        scenario_values[col] = ""
                    else:
                        scenario_values[col] = st.selectbox(
                            col,
                            unique_vals,
                            index=0,
                        )

            st.markdown("### Incentive Options for Bandit")
            candidate_incentives = st.multiselect(
                "Select which incentive columns should be considered as candidate 'arms'",
                mm.incentive_cols,
                default=mm.incentive_cols,
            )

            epsilon = st.slider(
                "Exploration ε (0 = pure exploit, higher = more exploration)",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
            )

            submitted = st.form_submit_button("Run Scenario")

        if submitted:
            # Build input row
            row = default_row.copy()
            for col, val in scenario_values.items():
                row[col] = val

            # Ensure numeric incentives; default 0 if not set
            for inc_col in mm.incentive_cols:
                if inc_col not in row.index:
                    row[inc_col] = 0
                elif not isinstance(row[inc_col], (int, float, np.number)):
                    # If came as categorical, force 0 (you can customise this)
                    try:
                        row[inc_col] = float(row[inc_col])
                    except Exception:
                        row[inc_col] = 0

            scenario_df = pd.DataFrame([row])

            try:
                preds = mm.predict_all(scenario_df)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                preds = {}

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Predicted Outcomes")
                st.write(preds)

            with col_right:
                if candidate_incentives:
                    try:
                        best_inc, reward, pred_A, pred_J = mm.recommend_incentives(
                            base_row=row,
                            incentive_options=candidate_incentives,
                            exploration_epsilon=epsilon,
                        )

                        st.subheader("Bandit Recommendation")
                        st.markdown(
                            f"""
**Recommended incentive:** `{best_inc}`  
**Expected reward (β_A * adoption + β_J * Just-CE):** {reward:.3f}  
**Predicted adoption (with {best_inc}=1):** {pred_A:.3f}  
**Predicted Just-CE (with {best_inc}=1):** {pred_J:.3f}
"""
                        )
                    except Exception as e:
                        st.error(f"Bandit recommendation error: {e}")
                else:
                    st.info("Select at least one incentive column as candidate arm.")


# -------------------------
# SELF-AUDIT TAB
# -------------------------

with tab_audit:
    st.header("Step 4 – Self-Audit & Drift Monitoring")

    mm: CollaborativeCEModelManager = st.session_state.get("model_manager", None)

    if mm is None or not mm.metrics_history:
        st.warning("No training metrics available yet. Train the models first.")
    else:
        metrics_df = pd.DataFrame(mm.metrics_history)
        metrics_df.index.name = "training_round"

        st.subheader("Training RMSE Over Time")
        st.line_chart(metrics_df)

        st.subheader("Latest Metrics Snapshot")
        st.json(mm.metrics_history[-1])

        st.markdown(
            """
**Interpretation (you can extend this logic):**

- If RMSE starts to rise over successive rounds, you may have **concept drift**.  
- You can plug in group-wise justice analysis (e.g. income, rural/urban) by
  computing metrics by subgroup and adding them here.  
- You can then:
  - Adjust model class / features  
  - Add fairness regularisation  
  - Trigger deeper diagnostics
"""
        )
