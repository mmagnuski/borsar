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

    def add_path(self, name, path, study=None, task=None, relative_to='main'):
        '''Add path to given study and task.

        Parameters
        ----------
        name : str
            Name of the path. This name is used to later get the path from
            Paths object.
        path : str
            Path to add.
        study : str | None
            Study name for which the path should be registered. If None, the
            first added study is chosen.
        task : str | None
            Task name for which the path should be registered. If ``None``, no
            specific task is used - that is task is ``""``. Because of this
            ``None`` nad ``""`` work the same way (no specific task).
        relative_to : str | bool
            Specifies the name of the path, which the current ``path`` is
            relative to. By default main path of given study and task is used.
            If there is no main path for given study-task combination, the main
            study path is used. If the path added should not be relative use
            ``relative_to=False``.
        '''
        study = self._check_set_study(study, msg='add paths')
        task = self._check_set_task(study, task)
        if isinstance(path, str):
            path = Path(path)

        # check if main pressent
        no_main = self.paths is None
        if not no_main:
            main, has_main = self._get_path('main', study, task,
                                            raise_error=False)
            no_main = not has_main

            if no_main and task:
                main, has_main = self._get_path('main', study, '',
                                                raise_error=False)
                no_main = not has_main

        if relative_to == 'main' and no_main and not (name == 'main'):
            # if not - throw an error
            msg = ("You need to define study main path before adding paths "
                   "to this study (all paths added to given study are by "
                   "default added relative to study main path). If you want "
                   "to add a path to the study that is not relavie but "
                   "absolute, use relative_to=False.")
            raise ValueError(msg)

        # resolve relative_to
        if relative_to and not relative_to == name:
            if relative_to == 'main':
                relative_path = main
            else:
                relative_path = self._get_path(relative_to, study, task)
            path = relative_path / path

        self._add_path(study, task, name, path)

    def _add_path(self, study, task, name, path):
        '''Add path to given study and task under specific name.'''
        if self.paths is None:
            colnames = ['study', 'task', 'name', 'path']
            self.paths = pd.DataFrame(columns=colnames)

        selected = self._find(name, study, task)
        if selected.shape[0] == 0:
            idx = self.paths.shape[0]
        else:
            idx = selected.index[0]
            msg = 'There is already path "{}" for study "{}", task "{}". '
            msg = msg.format(name, study, task) + "Overwriting."
            warn(msg, RuntimeWarning)

        self.paths.loc[idx, 'study'] = study
        self.paths.loc[idx, 'task'] = task
        self.paths.loc[idx, 'name'] = name
        self.paths.loc[idx, 'path'] = path

    def _find(self, name, study, task):
        '''Find path with specific name for given study and task.'''
        query_str = 'study == "{}" & task == "{}" & name == "{}"'
        selected = self.paths.query(query_str.format(study, task, name))
        return selected

    def _get_path(self, name, study, task, raise_error=True):
        '''Check if given path is present for specified study and task.
        Return the path if ``raise_error=True`` or return the path and
        information about it's presence if ``raise_error=False``.'''
        if self.paths is None:
            ispresent = False
        else:
            selected = self._find(name, study, task)
            ispresent = selected.shape[0] > 0

        if not ispresent:
            if raise_error:
                # CONSIDER: maybe add more detalis about what
                #           can and can't be found
                msg = 'Could not find path "{}" for study "{}"'
                msg.format(name, study)
                if task:
                    msg += ', task "{}".'
                    msg.format(task)

                raise ValueError(msg)
            else:
                path = None
        else:
            path = selected.path.iloc[0]

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
