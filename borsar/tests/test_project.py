import pytest
from borsar.project import Paths

# create tempdir structure first?
# when adding path and no such path: warn
# when retrieving path and no such path: error


def test_paths():
    pth = Paths()
    pth.register_study('study1')
    pth.add_path('main', r'temp/std1', study='study1')
    pth.add_path('test1', 'abc')
    pth.add_path('test2', 'abc', relative_to='main')
    pth.add_path('test3', 'abc', study='study1')
    pth.add_path('test4', r'temp/std1/abc', relative_to=False)

    # test1, test2, test3 and test4 should be the same path
    test1_path = str(pth._get_path('test1', 'study1', ''))
    test2_path = str(pth._get_path('test2', 'study1', ''))
    test3_path = str(pth._get_path('test3', 'study1', ''))
    test4_path = str(pth._get_path('test4', 'study1', ''))
    assert test1_path == test2_path == test3_path == test4_path

    # now add task
    pth.register_task('task1', study='study1')

    # add test1 path for task
    pth.add_path('test1', 'tsk1', task='task1')

    # add test1 path for study, task
    pth.add_path('test2', 'tsk1', study='study1', task='task1')

    # pth.add_path('main', study='study1', task='task1')
    # then add_path to study/task

    # test1 and test2 should be the same
    test1_path = str(pth._get_path('test1', 'study1', 'task1'))
    test2_path = str(pth._get_path('test2', 'study1', 'task1'))
    assert test1_path == test2_path

    # check overwriting
    with pytest.warns(RuntimeWarning, match='Overwriting'):
        pth.add_path('test2', 'tsk1b', study='study1', task='task1')

    new_test2_path = test2_path.replace('tsk1', 'tsk1b')
    assert str(pth._get_path('test2', 'study1', 'task1')) == new_test2_path

    # pth.register_study('study2', tasks=['a', 'b', 'c'])
