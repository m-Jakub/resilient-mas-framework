import os
import re

# Słownik zamienników (Regex) z uwzględnieniem src.apg
IMPORT_MAPPINGS = {
    # 1. Moduły z agents -> src.common
    r"\bagents\.api_abstraction\b": "src.common.api_abstraction",
    r"\bagents\.llm_retry\b": "src.common.llm_retry",
    r"\bagents\.ontology\b": "src.common.ontology",
    r"\bagents\.identity_graph\b": "src.common.identity_graph",
    r"\bagents\.philosopher_agents\b": "src.common.philosopher_agents",
    
    # 2. Moduły z agents -> src.hde
    r"\bagents\.baseline_tutor\b": "src.hde.baseline_tutor",
    r"\bagents\.enhanced_ontology\b": "src.hde.enhanced_ontology",
    r"\bagents\.speech_queue\b": "src.hde.speech_queue",
    r"\bagents\.team\b": "src.hde.team",
    r"\bagents\.sysar_calculator\b": "src.hde.sysar_calculator",
    r"\bagents\.conversation_logger\b": "src.hde.conversation_logger",
    r"\bagents\.legacy\.conversation\b": "src.hde.legacy_conversation",
    r"\bagents\.legacy\.socrates_moderator\b": "src.hde.socrates_moderator",
    r"\bagents\.legacy\.team_conversation\b": "src.hde.team_conversation",
    
    # 3. Moduły z agents -> src.kg_cfr
    r"\bagents\.aegis_orchestrator\b": "src.kg_cfr.aegis_orchestrator",
    r"\bagents\.aegis_drau_context\b": "src.kg_cfr.aegis_drau_context",
    r"\bagents\.aegis_logger\b": "src.kg_cfr.aegis_logger",
    r"\bagents\.counterfactual_reflection\b": "src.kg_cfr.counterfactual_reflection",
    r"\bagents\.policy_module\b": "src.kg_cfr.policy_module",
    r"\bagents\.system_dispatcher\b": "src.kg_cfr.system_dispatcher",
    r"\bagents\.turn_pipeline\b": "src.kg_cfr.turn_pipeline",
    
    # 4. Moduły z agents -> src.apg
    r"\bagents\.phase4_synthesis\b": "src.apg.phase4_synthesis",
    r"\bagents\.claim_segmenter\b": "src.apg.claim_segmenter",
    
    # 5. Naprawa pułapki relatywnych importów (np. w counterfactual_reflection.py)
    r"from\s+\.llm_retry\b": "from src.common.llm_retry",
    r"from\s+\.ontology\b": "from src.common.ontology",
    
    # 6. Globalne utils -> src.common.utils
    r"from\s+utils\b": "from src.common.utils",
    r"import\s+utils\b": "import src.common.utils",
    r"from\s+utils\.": "from src.common.utils.",
    r"import\s+utils\.": "import src.common.utils."
}

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content
    for pattern, replacement in IMPORT_MAPPINGS.items():
        new_content = re.sub(pattern, replacement, new_content)

    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"[Zaktualizowano importy] {filepath}")

def main():
    directories_to_scan = ["src", "scripts", "tests"]
    
    for directory in directories_to_scan:
        if not os.path.exists(directory):
            continue
            
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    process_file(filepath)
                    
    print("\nGotowe! Projekt został pomyślnie zreorganizowany, a importy wskazują na 'src.apg'.")

if __name__ == "__main__":
    main()