import types
import pytest
from pathlib import Path
from borsar.project import Paths

# create tempdir structure first?
# when adding path and no such path: warn
# when retrieving path and no such path: error


def test_paths():
    pth = Paths()

    # _check_set_study when there are no studies
    msg = 'You have to register a study first.'
    with pytest.raises(ValueError, match=msg):
        pth.add_path('ulala', 'ojoj')

    pth.register_study('study1')

    with pytest.raises(ValueError, match='You should define study main path'):
        pth.add_path('test0', 'abcd')

    with pytest.warns(RuntimeWarning, match='has been already registered'):
        pth.register_study('study1')

    # _get_path when pth.paths is None
    _, ispresent = pth._get('main', 'study1', '', raise_error=False)
    assert not ispresent

    pth.add_path('main', r'temp/std1', study='study1', validate=False)
    pth.add_path('test1', 'abc', validate=False)
    pth.add_path('test2', 'abc', relative_to='main', validate=False)
    pth.add_path('test3', 'abc', study='study1', validate=False)
    pth.add_path('test4', r'temp/std1/abc', relative_to=False, validate=False)

    # test1, test2, test3 and test4 should be the same path
    test1_path = str(pth._get('test1', 'study1', ''))
    test2_path = str(pth._get('test2', 'study1', ''))
    test3_path = str(pth._get('test3', 'study1', ''))
    test4_path = str(pth._get('test4', 'study1', ''))
    assert test1_path == test2_path == test3_path == test4_path

    # now add task
    pth.register_task('task1', study='study1')

    # add test1 path for task
    pth.add_path('test1', 'tsk1', task='task1', validate=False)

    # add test1 path for study, task
    pth.add_path('test2', 'tsk1', study='study1', task='task1', validate=False)

    # use relative_to=some_name
    pth.add_path('rodżer', 'another', relative_to='test2', validate=False)
    expected_path = Path('temp/std1/abc/another')
    assert pth._get('rodżer', 'study1', '') == expected_path

    pth.add_path('rodżer2', 'another', task='task1', relative_to='test2',
                 validate=False)
    expected_path = Path('temp/std1/tsk1/another')
    assert pth._get('rodżer2', 'study1', 'task1') == expected_path

    msg = 'Task "task1" has been already registered for study "study1".'
    with pytest.warns(RuntimeWarning, match=msg):
        pth.register_task('task1', study='study1')

    # pth.add_path('main', study='study1', task='task1')
    # then add_path to study/task

    # test1 and test2 should be the same
    test1_path = str(pth._get('test1', 'study1', 'task1'))
    test2_path = str(pth._get('test2', 'study1', 'task1'))
    assert test1_path == test2_path

    # now the same using get_path
    args = dict(study='study1', task='task1')
    test1_path = pth.get_path('test1', **args)
    test2_path = pth.get_path('test2', **args)
    assert test1_path == test2_path

    args['as_str'] = False
    test1_path = pth.get_path('test1', **args)
    test2_path2 = pth.get_path('test2', **args)
    assert test1_path == test2_path2

    # check overwriting
    with pytest.warns(RuntimeWarning, match='Overwriting'):
        pth.add_path('test2', 'tsk1b', study='study1', task='task1',
                     validate=False)

    new_test2_path = test2_path.replace('tsk1', 'tsk1b')
    assert str(pth._get('test2', 'study1', 'task1')) == new_test2_path

    pth.register_study('study2', tasks=['a', 'b', 'c'])
    assert 'study2' in pth.studies
    assert all([task in pth.tasks['study2'] for task in list('abc')])

    # _get_path with ispresent=False that raises error
    msg = 'Could not find path "sarna" for study "study1"'
    with pytest.raises(ValueError, match=msg):
        pth._get('sarna', 'study1', '')

    msg = 'Could not find path "sarna" for study "study1", task "task1"'
    with pytest.raises(ValueError, match=msg):
        pth._get('sarna', 'study1', 'task1')

    # _check_set_study when no such study
    msg = 'No study "study3" found'
    with pytest.raises(ValueError, match=msg):
        pth.add_path('ulala', 'ojoj', study='study3', task='a')

    # _check_set_task when no such task
    msg = 'No task "aa" found for study "study2"'
    with pytest.raises(ValueError, match=msg):
        pth.add_path('ulala', 'ojoj', study='study2', task='aa')


def test_paths_data():
    pth = Paths()
    pth.register_study('test_study')

    def read_fun(p, **args):
        assert isinstance(p, Paths)
        assert 'task' in args.keys()
        assert 'study' in args.keys()
        return [1, 2, 3]

    pth.register_data('data1', read_fun)
    data = pth.get_data('data1')
    assert data == [1, 2, 3]

    # pth.data has function
    assert isinstance(pth.data.loc[0, 'data'], types.FunctionType)

    # now with caching
    pth.register_data('data1', read_fun, cache=True)
    data = pth.get_data('data1')
    assert data == [1, 2, 3]
    assert pth.data.loc[0, 'data'] == [1, 2, 3]


def test_validate(tmp_path):
    study_dir = tmp_path / 'temp_study'
    study_dir.mkdir()

    task_data_dir = study_dir / 'data' / 'eeg' / 'task1'
    task_data_dir.mkdir(parents=True)

    pth = Paths()
    pth.register_study('study1', tasks=['task1'])
    pth.add_path('main', study_dir)
    pth.add_path('eeg', [study_dir / 'data' / 'eeg', 'justjoking'])
    pth.add_path('eeg', 'task1', task='task1', relative_to='eeg')

    assert pth.get_path('eeg', task='task1', as_str=False) == task_data_dir
    assert pth.get_path('eeg', as_str=False) == task_data_dir.parent

    with pytest.raises(ValueError, match='Could not find'):
        pth.add_path('test', ['ab', 'cd'], task='task1')
