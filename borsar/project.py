import os
import os.path as op
import pandas as pd


class Paths(object):
    def __init__(self):
        '''FIXME: DOCS'''
        self.studies = list()
        self.tasks = dict()
        self.paths = None

    def register_study(self, study):
        '''FIXME: DOCS'''
        if study not in self.studies:
            self.studies.append(study)
            self.tasks[study] = list()
        # else raise warning?

    def register_task(self, task, study=None):
        '''FIXME: DOCS'''
        study = self._check_set_study(study, msg='register tasks')

        if task not in self.tasks[study]:
            self.tasks[study] = task
        # else raise warning?


    def add_path(self, name, path, study=None, task=None, relative_to=None):
        '''FIXME: DOCS'''
        study = self._check_set_study(study, msg='add paths')

        if task is None:
            task = ""
        if relative_to is None:
            relative_to = 'main'

        # FIXME: add _validate_study_and_task for checking whether study and
        #        tasks are valid

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
            path = op.join(relative_path, path)

        self._add_path(study, task, name, path)

    def _add_path(self, study, task, name, path):
        '''FIXME: DOCS'''
        if self.paths is None:
            self.paths = pd.DataFrame(columns=['study', 'task', 'name', 'path'])

        idx = self.paths.shape[0]
        self.paths.loc[idx, 'study'] = study
        self.paths.loc[idx, 'task'] = task
        self.paths.loc[idx, 'name'] = name
        self.paths.loc[idx, 'path'] = path

    def _get_path(self, name, study, task, raise_error=True):
        '''Check if given path is present for specified study and task.
        Return the path if ``raise_error=True`` or return the path and
        information about it's presence if ``raise_error=False``.'''
        if self.paths is None:
            ispresent = False
        else:
            query_str = 'study == "{}" & task == "{}" & name == "{}"'
            selected = self.paths.query(query_str.format(study, task, name))
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
        return study

    # def _retrieve_path(self, name, study, task)
