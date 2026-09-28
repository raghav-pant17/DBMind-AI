"""DBMind-AI orchestration workflow."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from dbmind_ai.domain import DBMindResponse, ValidationResult, ValidationStatus
from dbmind_ai.graph.nodes import WorkflowNodes
from dbmind_ai.graph.state import DBMindState


# Orchestrates the DBMind-AI validation, correction, execution, and insight flow.
class DBMindWorkflow:
    """Run the DBMind-AI validation, correction, execution, and insight flow."""

    # Stores workflow nodes and execution thresholds.
    def __init__(
        self,
        nodes: WorkflowNodes,
        min_execution_score: int = 90,
        max_corrections: int = 2,
    ) -> None:
        self.nodes = nodes
        self.min_execution_score = min_execution_score
        self.max_corrections = max_corrections
        self.graph = self._build_graph()

    # Runs one user question through the complete DBMind-AI pipeline.
    def run(self, question: str) -> DBMindResponse:
        state = self.graph.invoke({"user_question": question, "correction_count": 0})
        return state["final_response"]

    # Builds the executable LangGraph state machine.
    def _build_graph(self):
        graph = StateGraph(DBMindState)
        graph.add_node("intent", self._intent_node)
        graph.add_node("schema_rag", self._schema_rag_node)
        graph.add_node("business_rules_rag", self._business_rules_rag_node)
        graph.add_node("generate_sql", self._generate_sql_node)
        graph.add_node("sql_guardrail", self._sql_guardrail_node)
        graph.add_node("join_validation", self._join_validation_node)
        graph.add_node("aggregation_validation", self._aggregation_validation_node)
        graph.add_node("business_validation", self._business_validation_node)
        graph.add_node("performance_validation", self._performance_validation_node)
        graph.add_node("aggregate_validations", self._aggregate_validations_node)
        graph.add_node("score", self._score_node)
        graph.add_node("correct_sql", self._correct_sql_node)
        graph.add_node("reject", self._reject_node)
        graph.add_node("execute_sql", self._execute_sql_node)
        graph.add_node("validate_result", self._validate_result_node)
        graph.add_node("insight", self._insight_node)

        graph.add_edge(START, "intent")
        graph.add_edge("intent", "schema_rag")
        graph.add_edge("schema_rag", "business_rules_rag")
        graph.add_edge("business_rules_rag", "generate_sql")
        graph.add_edge("generate_sql", "sql_guardrail")
        graph.add_conditional_edges(
            "sql_guardrail",
            self._route_after_guardrail,
            {
                "score": "score",
                "join_validation": "join_validation",
                "aggregation_validation": "aggregation_validation",
                "business_validation": "business_validation",
                "performance_validation": "performance_validation",
            },
        )
        graph.add_edge(
            [
                "join_validation",
                "aggregation_validation",
                "business_validation",
                "performance_validation",
            ],
            "aggregate_validations",
        )
        graph.add_edge("aggregate_validations", "score")
        graph.add_conditional_edges(
            "score",
            self._route_after_score,
            {
                "correct": "correct_sql",
                "reject": "reject",
                "execute": "execute_sql",
            },
        )
        graph.add_edge("correct_sql", "sql_guardrail")
        graph.add_edge("reject", END)
        graph.add_edge("execute_sql", "validate_result")
        graph.add_edge("validate_result", "insight")
        graph.add_edge("insight", END)
        return graph.compile()

    # Extracts intent and stores it in graph state.
    def _intent_node(self, state: DBMindState) -> DBMindState:
        return {"intent": self.nodes.intent_agent.run(state["user_question"])}

    # Retrieves schema context and stores it in graph state.
    def _schema_rag_node(self, state: DBMindState) -> DBMindState:
        return {
            "retrieved_schema": self.nodes.schema_retriever.retrieve(state["user_question"]),
            "full_schema": self.nodes.schema_retriever.all(),
        }

    # Retrieves business rules and stores them in graph state.
    def _business_rules_rag_node(self, state: DBMindState) -> DBMindState:
        return {"business_rules": self.nodes.rules_retriever.retrieve(state["user_question"])}

    # Generates the first SQL candidate from the current graph state.
    def _generate_sql_node(self, state: DBMindState) -> DBMindState:
        return {
            "generated_sql": self.nodes.sql_generator.run(
                state["user_question"],
                state["intent"],
                state["retrieved_schema"],
                state["business_rules"],
            )
        }

    # Runs the deterministic SQL safety gate before parallel validators.
    def _sql_guardrail_node(self, state: DBMindState) -> DBMindState:
        guardrail = self.nodes.sql_guardrail.validate(
            state["generated_sql"],
            state["full_schema"],
        )
        if guardrail.status == ValidationStatus.FAIL:
            return {
                "sql_guardrail_result": guardrail,
                "validations": [guardrail],
                "failures": [guardrail],
            }
        return {"sql_guardrail_result": guardrail}

    # Routes failed guardrails to scoring or valid SQL to parallel validators.
    def _route_after_guardrail(self, state: DBMindState) -> str | list[str]:
        if state["sql_guardrail_result"].status == ValidationStatus.FAIL:
            return "score"
        return [
            "join_validation",
            "aggregation_validation",
            "business_validation",
            "performance_validation",
        ]

    # Runs join validation as an independent LangGraph branch.
    def _join_validation_node(self, state: DBMindState) -> DBMindState:
        return {
            "join_validation": self.nodes.join_validator.run(
                state["generated_sql"],
                state["full_schema"],
            )
        }

    # Runs aggregation validation as an independent LangGraph branch.
    def _aggregation_validation_node(self, state: DBMindState) -> DBMindState:
        return {
            "aggregation_validation": self.nodes.aggregation_validator.run(
                question=state["user_question"],
                sql=state["generated_sql"],
                intent=state["intent"],
                schema=state["full_schema"],
                rules=state["business_rules"],
            )
        }

    # Runs business-rule validation as an independent LangGraph branch.
    def _business_validation_node(self, state: DBMindState) -> DBMindState:
        return {
            "business_validation": self.nodes.business_validator.run(
                question=state["user_question"],
                sql=state["generated_sql"],
                intent=state["intent"],
                rules=state["business_rules"],
                schema=state["full_schema"],
            )
        }

    # Runs performance validation as an independent LangGraph branch.
    def _performance_validation_node(self, state: DBMindState) -> DBMindState:
        return {"performance_validation": self.nodes.performance_agent.run(state["generated_sql"])}

    # Merges the guardrail result and parallel validator branch outputs.
    def _aggregate_validations_node(self, state: DBMindState) -> DBMindState:
        validations = [
            state["sql_guardrail_result"],
            state["join_validation"],
            state["aggregation_validation"],
            state["business_validation"],
            state["performance_validation"],
        ]
        failures = [result for result in validations if result.status == ValidationStatus.FAIL]
        return {"validations": validations, "failures": failures}

    # Calculates the trust score from current SQL validations.
    def _score_node(self, state: DBMindState) -> DBMindState:
        return {"trust_score": self._score(state["validations"])}

    # Routes graph execution to correction, rejection, or SQL execution.
    def _route_after_score(self, state: DBMindState) -> Literal["correct", "reject", "execute"]:
        if state["failures"] and state["correction_count"] < self.max_corrections:
            return "correct"
        if state["failures"] or state["trust_score"] < self.min_execution_score:
            return "reject"
        return "execute"

    # Regenerates SQL and increments the correction counter.
    def _correct_sql_node(self, state: DBMindState) -> DBMindState:
        sql = self.nodes.correction_agent.run(
            state["user_question"],
            state["intent"],
            state["retrieved_schema"],
            state["business_rules"],
            state["failures"],
        )
        return {
            "generated_sql": sql,
            "correction_count": state["correction_count"] + 1,
        }

    # Builds a safe non-executed response when trust requirements fail.
    def _reject_node(self, state: DBMindState) -> DBMindState:
        answer = "The generated SQL did not meet the trust threshold and was not executed."
        return {
            "final_answer": answer,
            "final_response": DBMindResponse(
                question=state["user_question"],
                sql=state["generated_sql"],
                answer=answer,
                trust_score=state["trust_score"],
                validations=state["validations"],
                corrections=state["correction_count"],
            ),
        }

    # Executes trusted SQL and stores the query result.
    def _execute_sql_node(self, state: DBMindState) -> DBMindState:
        return {"query_result": self.nodes.executor.execute(state["generated_sql"])}

    # Validates executed query results and updates the score.
    def _validate_result_node(self, state: DBMindState) -> DBMindState:
        result_validation = self.nodes.result_validator.run(state["query_result"])
        validations = [*state["validations"], result_validation]
        return {
            "result_validation": result_validation,
            "validations": validations,
            "trust_score": self._score(validations),
        }

    # Builds the final successful DBMind-AI response.
    def _insight_node(self, state: DBMindState) -> DBMindState:
        answer = self.nodes.insight_agent.run(
            state["user_question"],
            state["generated_sql"],
            state["query_result"],
            state["trust_score"],
            state["validations"],
            state["business_rules"],
        )
        return {
            "final_answer": answer,
            "final_response": DBMindResponse(
                question=state["user_question"],
                sql=state["generated_sql"],
                answer=answer,
                trust_score=state["trust_score"],
                validations=state["validations"],
                result=state["query_result"],
                corrections=state["correction_count"],
            ),
        }

    # Combines validation scores into one trust score.
    @staticmethod
    def _score(validations: list[ValidationResult]) -> int:
        if not validations:
            return 0
        return round(sum(result.score for result in validations) / len(validations))
