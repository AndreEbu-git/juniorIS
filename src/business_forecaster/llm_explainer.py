from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)


@dataclass
class ExplanationPayload:
    explanation_type: str
    target_variable: str
    summary: Dict[str, Any]
    metrics: Dict[str, float]
    top_features: List[str]
    scenario_summary: Optional[Dict[str, Any]] = None


class LLMExplanationService:
    """Generate optional AI explanations with a deterministic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_forecast_payload(self, result: Any, summary: Dict[str, Any]) -> ExplanationPayload:
        return ExplanationPayload(
            explanation_type="forecast",
            target_variable=summary["target_variable"],
            summary=summary,
            metrics=result.metrics,
            top_features=summary.get("top_features", [])
        )

    def build_scenario_payload(
        self,
        result: Any,
        summary: Dict[str, Any],
        scenario_result: Any
    ) -> ExplanationPayload:
        return ExplanationPayload(
            explanation_type="scenario",
            target_variable=summary["target_variable"],
            summary=summary,
            metrics=result.metrics,
            top_features=summary.get("top_features", []),
            scenario_summary=scenario_result.impact_summary
        )

    def generate_explanation(self, payload: ExplanationPayload) -> str:
        if not self.is_configured():
            return self._fallback_explanation(payload)

        try:
            from openai import OpenAI
        except ImportError:
            return (
                "OpenAI explanations are enabled in the app design, but the `openai` package is not "
                "installed in this environment yet. "
                f"{self._fallback_explanation(payload)}"
            )

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                input=self._build_prompt(payload)
            )
            return response.output_text.strip()
        except Exception as exc:
            return (
                f"AI explanation could not be generated right now ({exc}). "
                f"{self._fallback_explanation(payload)}"
            )

    def _build_prompt(self, payload: ExplanationPayload) -> str:
        return (
            "You are explaining business forecasting results to a small-business user.\n"
            "Use plain English, be specific, and avoid unnecessary jargon.\n\n"
            "Write five short sections:\n"
            "1. Overall takeaway\n"
            "2. What the forecast suggests\n"
            "3. How reliable the model appears based on the metrics\n"
            "4. Which factors seem to matter most\n"
            "5. If scenario data is present, explain the scenario impact in business terms\n\n"
            "Keep the explanation grounded only in the provided data.\n\n"
            f"Data:\n{json.dumps(asdict(payload), default=str, indent=2)}"
        )

    def _fallback_explanation(self, payload: ExplanationPayload) -> str:
        avg_change = float(payload.summary.get("average_forecast_change_pct", 0.0))
        forecast_mean = float(payload.summary.get("forecast_mean", 0.0))
        latest_actual = float(payload.summary.get("last_actual", 0.0))
        r2 = float(payload.metrics.get("test_r2", 0.0))
        mape = float(payload.metrics.get("test_mape", 0.0))

        if r2 >= 0.7:
            reliability = "The model fit looks strong"
        elif r2 >= 0.4:
            reliability = "The model fit looks moderate"
        else:
            reliability = "The model fit looks limited"

        if mape <= 10:
            error_text = "forecast error is relatively low"
        elif mape <= 20:
            error_text = "forecast error is moderate"
        else:
            error_text = "forecast error is fairly high"

        drivers = ", ".join(payload.top_features[:3]) if payload.top_features else "no clear top drivers"
        explanation = (
            f"The forecast for {payload.target_variable} averages {forecast_mean:,.2f}, "
            f"which is {avg_change:+.1f}% compared with the latest observed value of {latest_actual:,.2f}. "
            f"{reliability} and the expected {error_text} based on test metrics "
            f"(R^2={r2:.3f}, MAPE={mape:.1f}%). "
            f"The most influential features in this run were {drivers}."
        )

        if payload.scenario_summary:
            scenario_average = float(payload.scenario_summary.get("scenario_average", 0.0))
            average_change_pct = float(payload.scenario_summary.get("average_change_pct", 0.0))
            total_change_pct = float(payload.scenario_summary.get("total_change_pct", 0.0))
            explanation += (
                f" Under the scenario analysis, the average projected value changes to "
                f"{scenario_average:,.2f}, which is {average_change_pct:+.1f}% versus the base forecast, "
                f"and total projected performance changes by {total_change_pct:+.1f}%."
            )

        return explanation
