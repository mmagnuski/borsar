import pytest
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

    with pytest.raises(ValueError, match='You need to define study main path'):
        pth.add_path('test0', 'abcd')

    with pytest.warns(RuntimeWarning, match='has been already registered'):
        pth.register_study('study1')

    # _get_path when pth.paths is None
    _, ispresent = pth._get_path('main', 'study1', '', raise_error=False)
    assert not ispresent

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

    # use relative_to=some_name
    pth.add_path('rodżer', 'another', relative_to='test2')
    expected_path = r'temp\std1\abc\another'
    assert str(object=pth._get_path('rodżer', 'study1', '')) == expected_path

    pth.add_path('rodżer2', 'another', task='task1', relative_to='test2')
    expected_path = r'temp\std1\tsk1\another'
    assert (str(object=pth._get_path('rodżer2', 'study1', 'task1'))
            == expected_path)

    msg = 'Task "task1" has been already registered for study "study1".'
    with pytest.warns(RuntimeWarning, match=msg):
        pth.register_task('task1', study='study1')

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

    pth.register_study('study2', tasks=['a', 'b', 'c'])
    assert 'study2' in pth.studies
    assert all([task in pth.tasks['study2'] for task in list('abc')])

    # _get_path with ispresent=False that raises error
    msg = 'Could not find path "sarna" for study "study1"'
    with pytest.raises(ValueError, match=msg):
        pth._get_path('sarna', 'study1', '')


    # _check_set_study when no such study
    msg = 'No study "study3" found'
    with pytest.raises(ValueError, match=msg):
        pth.add_path('ulala', 'ojoj', study='study3', task='a')

    # _check_set_task when no such task
    msg = 'No task "aa" found for study "study2"'
    with pytest.raises(ValueError, match=msg):
        pth.add_path('ulala', 'ojoj', study='study2', task='aa')
