from ama_kbqa.orchestrator_agent.agent import OrchestratorAgent

orchestrator = OrchestratorAgent("orch", "session-123")
result = orchestrator.ask("Wer ist der Bruder von Barack Obama? ")
print(result)
