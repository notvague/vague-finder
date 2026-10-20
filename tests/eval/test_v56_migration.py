"""v56 재현 도구의 Context 별칭 복원·실패 판정. 실제 서버·모델 호출 없음."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[2] / "experiments/reranking/results_v56_attr_runtime_validation/migrate_to_server.py"
spec = importlib.util.spec_from_file_location("v56_migration", path)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)
ALIASES = {base + "__dev": base + "__dev__generation_test" for base in migration.CONTEXT_INDEX_FIELDS}


class Client:
    def __init__(self, aliases, apply=True):
        self.aliases = dict(aliases)
        self.updates = []
        self.apply = apply

    def get_aliases(self):
        return SimpleNamespace(aliases=[SimpleNamespace(alias_name=a, collection_name=c) for a, c in self.aliases.items()])

    def update_collection_aliases(self, change_aliases_operations):
        self.updates.append(change_aliases_operations)
        if self.apply:
            for operation in change_aliases_operations:
                alias = operation.create_alias
                self.aliases[alias.alias_name] = alias.collection_name


def test_restores_missing_context_aliases_and_is_idempotent():
    src, dst = Client(ALIASES), Client({})
    assert migration.restore_aliases(src, dst, list(ALIASES.values()))["aliases_ok"]
    assert dst.aliases == ALIASES
    migration.restore_aliases(src, dst, list(ALIASES.values()))
    assert len(dst.updates) == 1


def test_conflicting_alias_is_not_overwritten():
    dst = Client({next(iter(ALIASES)): "another_generation"})
    with pytest.raises(RuntimeError, match="conflicts"):
        migration.restore_aliases(Client(ALIASES), dst, list(ALIASES.values()))
    assert not dst.updates


def test_source_without_context_aliases_is_rejected():
    with pytest.raises(RuntimeError, match="원본 Context 별칭 없음"):
        migration.restore_aliases(Client({}), Client({}), list(ALIASES.values()))


def test_missing_target_is_rejected_before_mutation():
    dst = Client({})
    with pytest.raises(RuntimeError, match="missing_targets"):
        migration.restore_aliases(Client(ALIASES), dst, [])
    assert not dst.updates


def test_failed_alias_creation_cannot_report_success():
    with pytest.raises(RuntimeError, match="복원 후 검증 실패"):
        migration.restore_aliases(Client(ALIASES), Client({}, apply=False), list(ALIASES.values()))
