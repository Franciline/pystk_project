"""Run: PYTHONPATH=src python src/unit_tests/test_workflow_commands.py."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from main import collect_offline
from main.evaluate import checkpoint_method


if __name__ == '__main__':
    for setup, count in [('singlemap', 1), ('multimap', 5)]:
        with patch.object(sys, 'argv', ['collect_offline.py', '--setup', setup]), \
             patch.object(collect_offline, 'collect_dataset') as collect:
            collect_offline.main()
            collect.assert_called_once()
            config = collect.call_args.args[0]
            assert len(collect_offline.train_tracks(config)) == count
    with TemporaryDirectory() as temporary:
        checkpoint = Path(temporary)/'renamed.pt'
        # Detection is independent of the filename and a shared config's sections.
        torch.save({'model': {}, 'categories': {}}, checkpoint)
        assert checkpoint_method(checkpoint) == 'dagger'
        torch.save({'actor': {}, 'critic': {}}, checkpoint)
        assert checkpoint_method(checkpoint) == 'rlpd'
        torch.save({'unknown': {}}, checkpoint)
        try:
            checkpoint_method(checkpoint)
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown checkpoint accepted')
    print('Single/multimap collection is exclusive; checkpoint types detected from contents')
