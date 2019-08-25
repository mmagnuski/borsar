import os
import types
from warnings import warn
from pathlib import Path

import pandas as pd


class Paths(object):
    def __init__(self):
        '''Paths object allows for convenient storage and access to various
        study and task-level paths.
        '''
        self.studies = list()
        self.tasks = dict()
        self.paths = None
        self.data = None

    def register_study(self, study, tasks=None):
        '''Register study.

        Parameters
        ----------
        study : str
            Name of the study to register.
        tasks : None | list of str
            If list of string: allows to additionaly register tasks along
            study registration. If ``None`` (default) - no tasks are
            registered.
        '''
        if study not in self.studies:
            self.studies.append(study)
            self.tasks[study] = list()
        else:
            warn('Study "{}" has been already registered.'.format(study),
                 RuntimeWarning)

        if isinstance(tasks, list):
            for task in tasks:
                self.register_task(task, study=study)

    def register_task(self, task, study=None):
        '''Register task for given study.

        Parameters
        ----------
        task : str
            Name of the study to register.
        '''
        study = self._check_set_study(study, msg='register tasks')

        if task not in self.tasks[study]:
            self.tasks[study].append(task)
        else:
            msg = 'Task "{}" has been already registered for study "{}".'
            warn(msg.format(task, study), RuntimeWarning)

    def register_data(self, name, data, study=None, task=None, cache=False):
        '''Add data reading function to specific study and task.

        Parameters
        ----------
        name : str
            Name of the data. This name is used to later get the data from
            Paths object.
        data : function
            Function that reads and returns the data. The function has to
            take the Paths object as the first argument and take study and
            task as keyword arguments. Additional keyword arguments are
            allowed and passed to the reading function from
            ``Paths.get_data()``.
        study : str | None
            Study name for which the data should be registered. If ``None``,
            the first added study is chosen.
        task : str | None
            Task name for which the path should be registered. If ``None``, no
            specific task is used - that is task is ``""``. Because of this
            ``None`` and ``""`` work the same way (no specific task).
        cache : bool
            Whether to cache the read data. If ``True`` the data are stored in
            ``Paths`` object and do not have to be read again.
        '''
        study = self._check_set_study(study, msg='add paths')
        task = self._check_set_task(study, task)

        idx = self._add(study, task, name, data, to='data')
        self.data.loc[idx, 'cache'] = cache

    def add_path(self, name, path, study=None, task=None, relative_to=None,
                 validate=True):
        '''Add path to given study and task.

        Parameters
        ----------
        name : str
            Name of the path. This name is used to later get the path from
            Paths object.
        path : str
            Path to add.
        study : str | None
            Study name for which the path should be registered. If ``None``,
            the first added study is chosen.
        task : str | None
            Task name for which the path should be registered. If ``None``, no
            specific task is used - that is task is ``""``. Because of this
            ``None`` and ``""`` work the same way (no specific task).
        relative_to : str | bool
            Specifies the name of the path, which the current ``path`` is
            relative to. By default main path of given study and task is used.
            If there is no path that matches the name given in ``relative_to``
            for given study-task combination, the study-notask combination is
            used. If the path you are trying to add should not be treated as
            relative use ``relative_to=False``.
        validate : bool
            Whether to validate path correctness. If ``True`` (default) adding
            non-existent path will result in an error. When adding a list of
            paths with ``validate=True`` (default) only one path that exists in
            the file system will be chosen. This is useful when creating Paths
            object that would work well on various computers with different
            project paths structure.
        '''
        study = self._check_set_study(study, msg='add paths')
        task = self._check_set_task(study, task)

        if isinstance(path, list) and not validate:
            raise NotImplementedError('Passing multiple paths is not '
                                      'implemented for `validate=False`.')
        if isinstance(path, str):
            path = Path(path)
        elif isinstance(path, list):
            path = [Path(pth) for pth in path]

        if relative_to is None:
            relative_to = False if name == 'main' else 'main'

        # check if main pressent
        if relative_to:
            has_paths = self.paths is not None
            has_relpath = has_paths
            if has_paths:
                relpath, has_relpath = self._get(relative_to, study, task,
                                                 raise_error=False)

            if not has_relpath and has_paths and task:
                relpath, has_relpath = self._get(relative_to, study, '',
                                                 raise_error=False)

            if not has_relpath:
                if relative_to == 'main':
                    msg = ("You should define study main path before adding "
                           "paths to this study (all paths added to given "
                           "study are by default added relative to study "
                           "'main' path). If you want to add a path to the "
                           "study that is not relavie but absolute, use "
                           "`relative_to=False`.")
                else:
                    msg = ('Could not find the relative path "{}" for study'
                           '"{}" for neither task "{}" or no task.')
                    msg.format(relative_to, study, task)
                raise ValueError(msg)

            if isinstance(path, list):
                path = [relpath / pth for pth in path]
            else:
                path = relpath / path

        if validate:
            path = get_valid_path(path)

        self._add(study, task, name, path)

    # CONSIDER: allow for fetching unique paths without specifying
    #           study and task
    # CONSIDER: allow to set as_str in some Paths settings
    #           allow to get multiple paths at once
    def get_path(self, name, study=None, task=None, as_str=True):
        '''Get path corresponding to specified name for given study and task.

        Parameters
        ----------
        name : str
            Path name.
        study : str | None
            Study name. Uses the first registered study if ``None`` (default).
        task : str | None
            Task name. Uses the first registered task for given study
            if ``task`` is ``None`` (default).
        as_str : bool
            Whether to return the path as string (when ``True``) or as
            ``pathlib.Path`` object (when ``False``). ``True`` by default.

        Returns
        -------
        path : str | pathlib.Path
            Path corresponding to given name, study and task.
        '''
        study = self._check_set_study(study, msg='add paths')
        task = self._check_set_task(study, task)

        path = self._get(name, study, task)
        if as_str:
            path = str(path)
        return path

    def get_data(self, name, study=None, task=None, **kwargs):
        '''Get data registered for specific study and task.

        Parameters
        ----------
        name : str
            Data name.
        study : str | None
            Study name. Uses the first registered study if ``None`` (default).
        task : str | None
            Task name. Uses the first registered task for given study
            if ``task`` is ``None`` (default).

        Returns
        -------
        data : unknown
            The data that has been registered for given study and task.
        '''
        study = self._check_set_study(study, msg='add paths')
        task = self._check_set_task(study, task)

        idx = self._get(name, study, task, find_in='data')
        data = self.data.loc[idx, 'data']
        if isinstance(data, types.FunctionType):
            data = data(self, study=study, task=task, **kwargs)

        if self.data.loc[idx, 'cache']:
            self.data.at[idx, 'data'] = data

        return data

    def _add(self, study, task, name, obj, to='paths'):
        '''Add path to given study and task under specific name.'''
        df = self.paths if to == 'paths' else self.data

        if df is None:
            colnames = ['study', 'task', 'name']
            df = pd.DataFrame(columns=colnames)
            setattr(self, to, df)

        selected = self._find(name, study, task, find_in=to)
        if selected.shape[0] == 0:
            idx = df.shape[0]
        else:
            idx = selected.index[0]
            what = 'path' if to == 'paths' else 'data'
            msg = 'There is already {} "{}" for study "{}", task "{}". '
            msg = msg.format(what, name, study, task) + "Overwriting."
            warn(msg, RuntimeWarning)

        df.loc[idx, 'study'] = study
        df.loc[idx, 'task'] = task
        df.loc[idx, 'name'] = name
        if to == 'paths':
            df.loc[idx, 'path'] = obj
        else:
            df.loc[idx, 'data'] = obj
        return idx

    def _find(self, name, study, task, find_in='paths'):
        '''Find path with specific name for given study and task.'''
        query_str = 'study == "{}" & task == "{}" & name == "{}"'
        df = self.paths if find_in == 'paths' else self.data
        selected = df.query(query_str.format(study, task, name))
        return selected

    def _get(self, name, study, task, raise_error=True, find_in='paths'):
        '''Check if given path is present for specified study and task.
        Return the path if ``raise_error=True`` or return the path and
        information about it's presence if ``raise_error=False``.'''
        df = self.paths if find_in == 'paths' else self.data
        if df is None:
            ispresent = False
        else:
            selected = self._find(name, study, task, find_in=find_in)
            ispresent = selected.shape[0] > 0

        if not ispresent:
            if raise_error:
                # CONSIDER: maybe add more detalis about what
                #           can and can't be found
                what = 'path' if find_in == 'paths' else 'data'
                msg = 'Could not find {} "{}" for study "{}"'
                msg = msg.format(what, name, study)
                if task:
                    msg += ', task "{}".'
                    msg = msg.format(task)

                raise ValueError(msg)
            else:
                path = None
        else:
            # FIXME: both path and data should return idx
            path = (selected.path.iloc[0] if find_in == 'paths'
                    else selected.data.index[0])

        if raise_error:
            return path
        else:
            return path, ispresent

    def _check_set_study(self, study, msg=None):
        '''Check if study is present.'''
        if study is None:
            if len(self.studies) == 0:
                full_msg = 'You have to register a study first.'
                if msg is not None:
                    full_msg = ("You can't {} when no study is registered. "
                                + full_msg)
                    full_msg.format(msg)
                raise ValueError(full_msg)
            else:
                study = self.studies[0]
        else:
            # check if such study is present
            if study not in self.studies:
                full_msg = 'No study "{}" found.'
                raise ValueError(full_msg.format(study))
        return study

    def _check_set_task(self, study, task):
        '''Check if task is present.'''
        if task is None:
            task = ""

        # check if given task is present for this study
        if not task == "":
            has_task = task in self.tasks[study]
            if not has_task:
                # no task, throw error
                full_msg = ('No task "{}" found for study "{}". You have to '
                            'register this task first.')
                raise ValueError(full_msg.format(task, study))

        return task


def get_valid_path(paths):
    '''
    Select the first path that exists on current machine.

    Parameters
    ----------
    paths : str | pathlib.Path | list of str | list of pathlib.Path
        List of paths to check.

    Returns
    -------
    pth : str
        The first path that exists on current machine.
    '''
    if not isinstance(paths, list):
        paths = [paths]

    for pth in paths:
        if os.path.exists(pth):
            return pth

    msg = ('Could not find valid path. None of the following paths '
           'exists:\n{}')
    paths = '\n'.join([str(pth) for pth in paths])
    msg = msg.format(paths)
    raise ValueError(msg)
