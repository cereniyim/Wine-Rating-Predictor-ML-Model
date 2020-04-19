import os

from logging import getLogger

import luigi
import docker
from docker.errors import NotFound
from pathlib import Path

data_root = Path(os.getenv('PROJECT_ROOT')) / 'data_root'

CONTAINER_TASK_ENV = {}
CONTAINER_TASK_VOLUMES = {
    str(data_root): {
        'bind': '/usr/share/data/',
        'mode': 'rw'
    }
}
CONTAINER_TASK_NET = os.getenv('ORCHESTRATOR_NETWORK', 'code_challenge_default')


class ContainerNotFound(Exception):
    pass


class ContainerClient:

    def run_container(self, image, name, command, configuration):
        """Method used to submit/run a container.

        This method should return a reference to the contianer.

        Parameters
        ----------
        image: str
            image name to run
        name: str
            The name for this container
        command: str or list
            The command to run in the container.
        configuration: dict
            configuration like accepted by docker-py's run method see
            https://docker-py.readthedocs.io/en/stable/containers.html
            your client implementation might need to translate this into
            kwargs accepted by your execution engine in this method.
        Returns
        -------
            container: obj
                container reference
        """
        raise NotImplementedError()

    def log_generator(self, container):
        """Generator to log stream.

        This method can return a generator to the containers log stream.
        If implemented the container logs will show up in the central scheduler
        as well as in the controller log. Generator is supposed to return lines
        as bytes.

        Parameters
        ----------
        container: contianer reference
            A container reference as returned from run_container method

        Returns
        -------
            log_stream: generator[bytes]
        """
        return None

    def get_exit_info(self, container):
        """Retrieve container exit status

        This method should return a tuple like (exit_code, exit_message). In
        case the container is still running it should block until exit
        information is available.

        Parameters
        ----------
        container: container reference
            A container reference as returned from run_container method

        Returns
        -------
            exit_info: tuple
        """
        raise NotImplementedError()

    def remove_container(self, container):
        """Remove a container from the execution engine.

        This method should remove the container and will be called
        only if the container finished successfully.

        Parameters
        ----------
        container: contianer reference
            A container reference as returned from run_container method

        Returns
        -------
            None
        """
        raise NotImplementedError()

    def stop_container(self, container):
        """Method to stop a running container

        Parameters
        ----------
        container: contianer reference
            A container reference as returned from run_container method

        Returns
        -------
            None
        """
        raise NotImplementedError()

    def get_container(self, u_name):
        """Retrieve container reference from name.

        Parameters
        ----------
        u_name: str
            unique container name or id usually contains retry suffix.

        Returns
        -------
            container: obj
                container reference
        """
        raise NotImplementedError()

    def get_executions(self, task_id):
        """Retrieve all previous runs of this task.

        The return value should be a list of container or job instances. It
        should be ordered by ascending retry count such that the last object
        is the most recently run.

        Parameters
        ----------
        task_id: str
            unique task name usually based on parameter hash.

        Returns
        -------
            list: list[container]
                list of container or job objects
        """
        raise NotImplementedError()

    def get_retry_count(self, task_id):
        """Get number of retries of this task from container engine.

        If last execution is still running it should return it's number else
        it should return the number of failed executions + 1

        Parameters
        ----------
        task_id: str
            unique task name usually based on parameter hash.

        Returns
        -------
            retry_count: int
        """
        raise NotImplementedError()


class DockerClient(ContainerClient):

    def __init__(self):
        self._c = docker.from_env()

    def run_container(self, image, name, command, configuration):
        try:
            container = self._c.containers.run(
                image=image,
                name=name,
                command=command,
                detach=True,
                **configuration)
        except docker.errors.APIError as e:
            if e.status_code == 409:
                log = getLogger(__name__)
                log.warning('Received API exception 409.')
                ltid = configuration['labels']['luigi_task_id']
                container = self.get_executions(ltid)
                log.info('Found existing container for this task. '
                         'Will try to reconnect')
            else:
                raise e
        return container

    def log_generator(self, container):
        return container.logs(stream=True)

    def get_exit_info(self, container):
        exit_info = container.wait()
        return exit_info['StatusCode'], exit_info['Error']

    def remove_container(self, container):
        container.remove(force=True)

    def stop_container(self, container):
        container.stop()

    def get_container(self, u_name):
        try:
            self._c.containers.get(u_name)
        except NotFound:
            raise ContainerNotFound()

    def get_executions(self, task_id):
        containers = self._c.containers.list(
            filters={'label': f'luigi_task_id={task_id}'},
            all=True,
        )
        containers = sorted(containers, key=lambda x: x.labels['luigi_retries'])
        return containers

    def get_retry_count(self, task_id):
        try:
            containers = self.get_executions(task_id)
            container = containers[-1]
            current = int(container.labels['luigi_retries'])
            if container.status != 'exited':
                return current
            else:
                return current + 1
        except IndexError:
            return 0


class ContainerTask(luigi.Task):
    """Base class to run containers."""

    no_remove_finished = luigi.BoolParameter(
        default=False, description="Don't remove containers "
                                   "that finished successfully.")

    CLIENT = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = self.get_client()
        self._container = None
        self.u_name = None
        self._retry_count = 0
        self._log = []

    def get_client(self)->ContainerClient:
        """Retrieve execution engine client.

        The client object will be saved to the private attribute `_client`.
        Override this method in case your client needs some special
        initialization.

        Returns
        -------
            client: object
                client to manage containers
        """
        return self.CLIENT()

    def get_logs(self):
        """Return container logs.

        This method returns container logs as a list of lines.
        It will only work if `log_stream` method is implemented.

        It is especially useful if a task does not create a file in a
        filesystem e.g. it might do some database operation in which
        case the log can be written to a file which would also serve
        as a flag that the task finished successfully. For this case
        you will have to override the `run` method to save the logs
        in the end.

        Returns
        -------
            logs: list of str
        """
        return self._log

    @property
    def name(self):
        """We name the resource with luigi's id.

        This id is based on a hash of a tasks parameters and helps avoid running
        the same task twice. If a task with this names already exists and failed
        it will append a 'retry-<NUMBER>' to the name.
        """
        task_id = self.task_id
        if len(task_id) > 53:
            name_components = task_id.split('_')
            name, param_hash = name_components[0], name_components[-1]
            return '-'.join([name[:43], param_hash]).lower()
        else:
            return task_id.lower().replace('_', '-')

    @property
    def command(self):
        """The command to be executed by the container."""
        raise NotImplementedError("Docker task must specify command")

    @property
    def image(self):
        """Which image to use to create the container."""
        raise NotImplementedError("Docker tasks must specify image")

    @property
    def labels(self):
        return dict(luigi_retries=str(self._retry_count),
                    luigi_task_id=self.name)

    @property
    def configuration(self):
        """Container configuration dictionary.

        Should return a dictionary as accepted by docker-py's run method
        see https://docker-py.readthedocs.io/en/stable/containers.html for more
        information which keys are accepted.

        It should be translated into other execution engines format by the
        corresponding subclass.
        """
        default = {'labels': self.labels}
        default.update({
            'environment': CONTAINER_TASK_ENV
        })
        default.update({
            'volumes': CONTAINER_TASK_VOLUMES
        })
        default.update({
            'network': CONTAINER_TASK_NET
        })
        return default

    def _set_name(self):
        if self._retry_count:
            self.u_name = '{}-retry-{}' \
                .format(self.name, self._retry_count + 1)
        else:
            self.u_name = self.name

    def run(self):
        """Actually submit and run task as a container."""
        try:
            self._run_and_track_task()
        finally:
            if self._container:
                self._client.stop_container(self._container)

    def _run_and_track_task(self):
        self._retry_count = self._client.get_retry_count(self.name)
        self._set_name()
        self._container = self._client.run_container(
            self.image,
            self.u_name,
            self.command,
            self.configuration,
        )
        self._log = []
        log_stream = self._client.log_generator(self._container)
        if log_stream is not None:
            for line in log_stream:
                self._log.append(line.decode().strip())
                getLogger('luigi-interface').info(self._log[-1])
                self.set_status_message('\n'.join(self._log))
        exit_info = self._client.get_exit_info(self._container)
        if exit_info[0] == 0:
            if not self.no_remove_finished:
                self._client.remove_container(self._container)
            self._container = None
        else:
            raise RuntimeError(
                "Container exited with status code:"
                " {} and msg: {}".format(exit_info[0],
                                         exit_info[1]))


class DockerTask(ContainerTask):
    """Run tasks as containers on a local docker engine."""

    CLIENT = DockerClient
