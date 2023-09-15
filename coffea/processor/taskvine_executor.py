import os
import re
import signal

from tempfile import NamedTemporaryFile, TemporaryDirectory
from os.path import join

import collections

import math

from coffea.util import rich_bar

import cloudpickle

from .executor import (
    WorkItem,
    _compression_wrapper,
    _decompress,
)

from .accumulator import (
    accumulate,
)


# The Work Queue object is global b/c we want to
# retain state between runs of the executor, such
# as connections to workers, cached data, etc.
manager = None

# If set to True, workflow stops processing and outputs only the results that
# have been already processed.
early_terminate = False


# This function that accumulates results from files does not require taskvine.
# We declare it before checking for taskvine so that we do not need to install taskvine at
# the remote site.
def accumulate_result_files(files_to_accumulate, accumulator=None):
    import time
    from datetime import datetime

    import cProfile
    import pstats
    import io
    from pstats import SortKey

    pr = cProfile.Profile()
    import numpy as np

    start = datetime.now().strftime("%H_%M_%S")

    from coffea.processor import accumulate

    # work on local copy of list
    files_to_accumulate = list(files_to_accumulate)

    load_time = 0
    accum_time = 0

    pr.enable()
    while files_to_accumulate:
        f = files_to_accumulate.pop()

        t = time.time_ns()
        with open(f, "rb") as rf:
            result = _decompress(rf.read())
            load_time += time.time_ns() - t

        if not accumulator:
            accumulator = result
            continue

        t = time.time_ns()
        accumulator = accumulate([result], accumulator)
        accum_time += time.time_ns() - t

        del result

    end = datetime.now().strftime("%H_%M_%S")

    print(
        f"--------------- ACCUMTIMES: START: {start} END: {end} LOAD: {load_time/1e9} MERGE: {accum_time/1e9}",
        flush=True,
    )
    pr.disable()

    for k, h in accumulator["out"].items():
        vs = h.view(flow=True)
        total = 0
        zeros = 0

        for v in vs.values():
            total += len(v.ravel())
            zeros += np.sum(v == 0)
        print(
            f"{k}  zeros %: {zeros}/{total} {100.0*(zeros/max(total,1)):6.2f}  total size ~GB: {8.0*total/(1024.0*1024.0*1024.0)}"
        )

    print("--- cumulative profile:")
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())
    print("--- time profile:")
    s = io.StringIO()
    sortby = SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())

    print("++++++++++++++")

    return accumulator


try:
    import ndcctools.taskvine as vine
    from ndcctools.taskvine import Manager, PythonTask, PythonTaskNoResult
except ImportError:
    wq = None
    print("ndcctools.taskvine module not available")

    class PythonTask:
        def __init__(self, *args, **kwargs):
            raise ImportError("ndcctools.taskvine not available")

    class Manager:
        def __init__(self, *args, **kwargs):
            raise ImportError("ndcctools.taskvine not available")


TaskReport = collections.namedtuple(
    "TaskReport", ["events_count", "wall_time", "memory"]
)


class CoffeaVine(Manager):
    def __init__(
        self,
        executor,
    ):
        self._staging_dir_obj = TemporaryDirectory(prefix="vine-tmp-", dir=executor.filepath)

        self.executor = executor

        self.stats_coffea = Stats()

        # set to keep track of the final work items the workflow consists of.
        # When a work item needs to be split, it is replaced from this set by
        # its constituents.
        self.known_workitems = set()

        # list that keeps results as they finish to construct accumulation
        # tasks.
        self.tasks_to_accumulate = []

        # list of TaskReport tuples with the statistics to compute the dynamic
        # chunksize
        self.task_reports = []

        super().__init__(
            port=self.executor.port,
            name=self.executor.manager_name,
            staging_path=self._staging_dir_obj.name,
            status_display_interval=self.executor.status_display_interval,
            ssl=self.executor.ssl,
        )

        self.extra_input_files = {
            name: self.declare_file(name, cache=True)
            for name in executor.extra_input_files
        }

        if self.executor.x509_proxy:
            self.extra_input_files["x509.pem"] = self.declare_file(executor.x509_proxy, cache=True)

        self.poncho_file = False
        if self.executor.environment_file:
            self.poncho_file = self.declare_poncho(executor.environment_file, cache=True)

        self.bar = StatusBar(enabled=executor.status)
        self.console = VerbosePrint(self.bar.console, executor.status, executor.verbose)

        self._declare_resources()

        # Make use of the stored password file, if enabled.
        if self.executor.password_file:
            self.set_password_file(self.executor.password_file)

        self.console.printf(f"Listening for TaskVine workers on port {self.port}.")
        # perform a wait to print any warnings before progress bars
        self.wait(0)

    def __del__(self):
        try:
            self._staging_dir_obj.cleanup()
        except Exception:
            pass

    def _check_executor_parameters(self, executor):
        if executor.treereduction < 2:
            raise ValueError("TaskVineExecutor: treereduction should be at least 2.")

        if not executor.port:
            executor.port = 0 if executor.manager_name else 9123

        # taskvine always needs serializaiton to files, thus compression is always on
        if executor.compression is None:
            executor.compression = 1

        # activate monitoring if it has not been explicitely activated and we are
        # using an automatic resource allocation.
        if executor.resources_mode != "fixed" and executor.resource_monitor == "off":
            executor.resource_monitor = "watchdog"

        executor.verbose = executor.verbose or executor.print_stdout
        executor.x509_proxy = _get_x509_proxy(executor.x509_proxy)

    def submit(self, task):
        taskid = super().submit(task)
        self.console(
            "submitted {category} task id {id} item {item}, with {size} {unit}(s)",
            category=task.category,
            id=taskid,
            item=task.itemid,
            size=len(task),
            unit=self.executor.unit,
        )
        return taskid

    def wait(self, timeout=None):
        task = super().wait(timeout)
        if task:
            task.report(self)
            # Evaluate and display details of the completed task
            if task.successful():
                self.stats_coffea.max(
                    "bytes_received", task.get_metric("bytes_received") / 1e6
                )
                self.stats_coffea.max("bytes_sent", task.get_metric("bytes_sent") / 1e6)
                # Remove input files as we go to avoid unbounded disk we do not
                # remove outputs, as they are used by further accumulate tasks
                task.cleanup_inputs(self)
            return task
        return None

    def application_info(self):
        return {
            "application_info": {
                "values": dict(self.stats_coffea),
                "units": {
                    "bytes_received": "MB",
                    "bytes_sent": "MB",
                },
            }
        }

    @property
    def staging_dir(self):
        return self._staging_dir_obj.name

    @property
    def chunksize_current(self):
        return self._chunksize_current

    @chunksize_current.setter
    def chunksize_current(self, new_value):
        self._chunksize_current = new_value
        self.stats_coffea["chunksize_current"] = self._chunksize_current

    @property
    def executor(self):
        return self._executor

    @executor.setter
    def executor(self, new_value):
        self._check_executor_parameters(new_value)
        self._executor = new_value

    def function_to_file(self, function, name=None):
        with NamedTemporaryFile(
            prefix=name, suffix=".p", dir=self.staging_dir, delete=False
        ) as f:
            cloudpickle.dump(function, f)
            return f.name

    def soft_terminate(self, task=None):
        if task:
            self.console.warn(f"item {task.itemid} failed permanently.")

        if not early_terminate:
            # trigger soft termination
            _handle_early_terminate(0, None, raise_on_repeat=False)

    def _add_task_report(self, task):
        r = TaskReport(
            len(task), task.get_metric("time_workers_execute_last") / 1e6, task.resources_measured.memory
        )
        self.task_reports.append(r)

    def _preprocessing(self, items, function, accumulator):
        function = _compression_wrapper(self.executor.compression, function)
        for item in items:
            task = PreProcTask(self, function, item)
            self.submit(task)

        self.bar.add_task("Preprocessing", total=len(items), unit=self.executor.unit)
        while not self.empty():
            task = self.wait(5)
            if task:
                if task.successful():
                    accumulator = accumulate([task.output], accumulator)
                    self.bar.advance("Preprocessing", 1)
                    task.cleanup_outputs(self)
                else:
                    task.resubmit(self)
                self.bar.refresh()

        self.bar.stop_task("Preprocessing")
        return accumulator

    def _submit_processing_tasks(self, proc_fn, items):
        while True:
            if early_terminate or self._items_empty:
                return
            sc = self.stats_coffea
            if sc["events_submitted"] >= sc["events_total"]:
                return
            if not self.hungry():
                return
            try:
                if sc["events_submitted"] > 0:
                    # can't send if generator not initialized first with a next
                    item = items.send(self.chunksize_current)
                else:
                    item = next(items)
                self._submit_processing_task(proc_fn, item)
            except StopIteration:
                self.console.warn("Ran out of items to process.")
                self._items_empty = True
                return

    def _submit_processing_task(self, proc_fn, item):
        self.known_workitems.add(item)
        t = ProcTask(self, proc_fn, item)
        self.submit(t)
        self.stats_coffea["events_submitted"] += len(t)

    def _final_accumulation(self, accumulator):
        if len(self.tasks_to_accumulate) < 1:
            self.console.warn("No results available.")
            return accumulator

        self.console("Merging with local final accumulator...")
        accumulator = accumulate_result_files(
            [t.output_file.source() for t in self.tasks_to_accumulate], accumulator
        )

        total_accumulated_events = 0
        for t in self.tasks_to_accumulate:
            total_accumulated_events += len(t)
            t.cleanup_outputs(self)

        sc = self.stats_coffea
        if sc["events_processed"] != sc["events_total"]:
            self.console.warn(
                f"Number of events processed ({sc['events_processed']}) is different from total ({sc['events_total']})!"
            )

        if total_accumulated_events != sc["events_processed"]:
            self.console.warn(
                f"Number of events accumulated ({total_accumulated_events}) is different from processed ({sc['events_processed']})!"
            )

        return accumulator

    def _fill_unprocessed_items(self, accumulator, items):
        chunksize = max(self.chunksize_current, self.executor.chunksize)
        try:
            while True:
                if chunksize != self.executor.chunksize:
                    item = items.send(chunksize)
                else:
                    item = next(items)
                self.known_workitems.add(item)
        except StopIteration:
            pass

        if accumulator:
            unproc = self.known_workitems - accumulator["processed"]
            accumulator["unprocessed"] = unproc
            if unproc:
                count = sum(len(item) for item in unproc)
                self.console.warn(f"{len(unproc)} unprocessed item(s) ({count} event(s)).")

    def _processing(self, items, function, accumulator):
        function = _compression_wrapper(self.executor.compression, function)
        accumulate_fn = _compression_wrapper(
            self.executor.compression, accumulate_result_files
        )

        executor = self.executor
        sc = self.stats_coffea

        # Ensure that the items looks like a generator
        if not isinstance(items, collections.abc.Generator):
            items = (item for item in items)

        # Keep track of total tasks in each state.
        sc["events_processed"] = 0
        sc["events_submitted"] = 0
        sc["events_total"] = executor.events_total
        sc["accumulations_submitted"] = 0
        sc["chunksize_original"] = executor.chunksize

        self.chunksize_current = executor.chunksize

        self._make_process_bars()

        signal.signal(signal.SIGINT, _handle_early_terminate)

        self._process_events(function, accumulate_fn, items)

        # merge results with original accumulator given by the executor
        accumulator = self._final_accumulation(accumulator)

        # compute the set of unprocessed work items, if any
        self._fill_unprocessed_items(accumulator, items)

        if self.chunksize_current != sc["chunksize_original"]:
            self.console.printf(f"final chunksize {self.chunksize_current}")

        self._update_bars(final_update=True)
        return accumulator

    def _process_events(self, proc_fn, accum_fn, items):
        self.known_workitems = set()
        sc = self.stats_coffea
        self._items_empty = False

        while True:
            if self.empty():
                if self._items_empty:
                    break
                if sc["events_total"] <= sc["events_processed"] and len(self.tasks_to_accumulate) < 2:
                    break

            self._submit_processing_tasks(proc_fn, items)

            # When done submitting, look for completed tasks.
            task = self.wait(5)
            if task:
                if not task.successful():
                    task.resubmit(self)
                    continue
                self.tasks_to_accumulate.append(task)
                if re.match("processing", task.category):
                    self._add_task_report(task)
                    sc["events_processed"] += len(task)
                    sc["chunks_processed"] += 1
                elif task.category == "accumulating":
                    sc["accumulations_done"] += 1
                else:
                    raise RuntimeError(f"Unrecognized task category {task.category}")

            self._submit_accum_tasks(accum_fn)
            self._update_bars()

    def _submit_accum_tasks(self, accum_fn):
        treereduction = self.executor.treereduction

        sc = self.stats_coffea
        bring_back = False
        force = False
        min_accum = treereduction

        if sc["events_processed"] >= sc["events_total"]  or early_terminate:
            s = self.stats
            if s.tasks_waiting + s.tasks_on_workers == 0:
                force = True
                if len(self.tasks_to_accumulate) < 2:
                    return
                elif min_accum > len(self.tasks_to_accumulate):
                    bring_back = True
                    min_accum = 1

        if len(self.tasks_to_accumulate) < (2 * treereduction) - 1 and (not force):
            return

        self.tasks_to_accumulate.sort(key=lambda t: len(t))
        for start in range(0, len(self.tasks_to_accumulate), treereduction):
            if len(self.tasks_to_accumulate) < min_accum:
                break

            end = min(len(self.tasks_to_accumulate), treereduction)
            next_to_accum = self.tasks_to_accumulate[0:end]
            self.tasks_to_accumulate = self.tasks_to_accumulate[end:]

            accum_task = AccumTask(self, accum_fn, next_to_accum, bring_back_output=bring_back)
            self.submit(accum_task)
            sc["accumulations_submitted"] += 1

    def _declare_resources(self):
        executor = self.executor

        # If explicit resources are given, collect them into default_resources
        default_resources = {}
        if executor.cores:
            default_resources["cores"] = executor.cores
        if executor.memory:
            default_resources["memory"] = executor.memory
        if executor.disk:
            default_resources["disk"] = executor.disk
        if executor.gpus:
            default_resources["gpus"] = executor.gpus

        # Enable monitoring and auto resource consumption, if desired:
        self.tune("category-steady-n-tasks", 3)
        self.tune("category-steady-n-tasks", 1)

        # Evenly divide resources in workers per category
        self.tune("force-proportional-resources", 1)

        self.tune("hungry-minimum", 100)

        # if resource_monitor is given, and not 'off', then monitoring is activated.
        # anything other than 'measure' is assumed to be 'watchdog' mode, where in
        # addition to measuring resources, tasks are killed if they go over their
        # resources.
        monitor_enabled = True
        watchdog_enabled = True
        if not executor.resource_monitor or executor.resource_monitor == "off":
            monitor_enabled = False
        elif executor.resource_monitor == "measure":
            watchdog_enabled = False

        if monitor_enabled:
            self.enable_monitoring(watchdog=watchdog_enabled)

        # set the auto resource modes
        mode = "max"
        if executor.resources_mode == "fixed":
            mode = "fixed"
        for category in "default preprocessing processing accumulating".split():
            self.set_category_mode(category, mode)
            # self.set_category_resources_max(category, default_resources)

        # use auto mode max-throughput only for processing tasks
        if executor.resources_mode == "max-throughput":
            self.set_category_mode("processing", "max throughput")

        # enable fast termination of workers
        fast_terminate = executor.fast_terminate_workers
        for category in "default preprocessing processing accumulating".split():
            if fast_terminate and fast_terminate > 1:
                self.activate_fast_abort_category(category, fast_terminate)

    def _make_process_bars(self):
        accums = self._estimate_accum_tasks()

        self.bar.add_task(
            "Submitted", total=self.executor.events_total, unit=self.executor.unit
        )
        self.bar.add_task(
            "Processed", total=self.executor.events_total, unit=self.executor.unit
        )
        self.bar.add_task("Accumulated", total=math.ceil(accums), unit="tasks")

        self.stats_coffea["chunks_processed"] = 0
        self.stats_coffea["accumulations_done"] = 0
        self.stats_coffea["accumulations_submitted"] = 0
        self.stats_coffea["estimated_total_accumulations"] = accums

        self._update_bars()

    def _estimate_accum_tasks(self):
        sc = self.stats_coffea

        try:
            # return immediately if there is no more work to do
            if sc["events_total"] <= sc["events_processed"]:
                if sc["accumulations_submitted"] <= sc["accumulations_done"]:
                    return sc["accumulations_done"]

            items_to_accum = sc["chunks_processed"]
            items_to_accum += sc["accumulations_submitted"]

            events_left = sc["events_total"] - sc["events_processed"]
            chunks_left = math.ceil(events_left / sc["chunksize_current"])
            items_to_accum += chunks_left

            accums = 1
            while True:
                if items_to_accum <= self.executor.treereduction:
                    accums += 1
                    break
                step = math.floor(items_to_accum / self.executor.treereduction)
                accums += step
                items_to_accum -= step * self.executor.treereduction
            return accums
        except Exception:
            return 0

    def _update_bars(self, final_update=False):
        sc = self.stats_coffea
        total = sc["events_total"]

        accums = self._estimate_accum_tasks()

        self.bar.update("Submitted", completed=sc["events_submitted"], total=total)
        self.bar.update("Processed", completed=sc["events_processed"], total=total)
        self.bar.update("Accumulated", completed=sc["accumulations_done"], total=accums)

        sc["estimated_total_accumulations"] = accums

        self.bar.refresh()
        if final_update:
            self.bar.stop()


class CoffeaVineTask(PythonTask):
    tasks_counter = 0

    def __init__(self, m, fn, item_args, itemid, bring_back_output=False):
        CoffeaVineTask.tasks_counter += 1
        self.itemid = itemid
        self.retries_to_go = m.executor.retries
        self.function = fn
        super().__init__(self.function, *item_args)

        self.disable_output_serialization()

        if not bring_back_output:
            self.enable_temp_output()

        for name, f in m.extra_input_files.items():
            self.add_input(f, name)

        if m.executor.x509_proxy:
            self.set_env_var('X509_USER_PROXY', "x509.pem")

        if m.poncho_file:
            self.add_environment(m.poncho_file)

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.itemid)

    def _has_result(self):
        return not isinstance(self.output, PythonTaskNoResult)

    # use output to return python result, rather than stdout as regular wq
    @property
    def output(self):
        if not self._output_loaded:
            if self.successful():
                try:
                    with open(os.path.join(self._tmpdir, self._out_name_file), "rb") as rf:
                        self._output = _decompress(rf.read())
                except Exception as e:
                    raise e
                    self._output = ResultUnavailable(e)
            else:
                self._output = PythonTaskNoResult()
                print(self.std_output)
            self._output_loaded = True
        return self._output

    def cleanup_inputs(self, m):
        pass

    def cleanup_outputs(self, m):
        m.remove_file(self.output_file)

    def clone(self, m):
        raise NotImplementedError

    def resubmit(self, m):
        if self.retries_to_go < 1:
            return m.soft_terminate(self)

        t = self.clone(m)
        t.retries_to_go = self.retries_to_go - 1

        m.console(
            "resubmitting {} as {} with {} events. {} attempt(s) left.",
            self.itemid,
            t.itemid,
            len(t),
            t.retries_to_go,
        )

        m.submit(t)

    def split(self, m):
        # if tasks do not overwrite this method, then is is assumed they cannot
        # be split.
        m.soft_terminate(self)

    def debug_info(self):
        msg = "{} with '{}' result.".format(self.itemid, self.result)
        return msg

    def exhausted(self):
        return self.result == "resource exhaustion"

    def report(self, m):
        if (not m.console.verbose_mode) and self.successful():
            return self.successful()

        m.console.printf(
            "{} task id {} item {} with {} events on {}. return code {} ({})",
            self.category,
            self.id,
            self.itemid,
            len(self),
            self.hostname,
            self.exit_code,
            self.result,
        )

        m.console.printf(
            "    allocated cores: {:.1f}, memory: {:.0f} MB, disk {:.0f} MB, gpus: {:.1f}",
            self.resources_allocated.cores,
            self.resources_allocated.memory,
            self.resources_allocated.disk,
            self.resources_allocated.gpus,
        )

        if m.executor.resource_monitor and m.executor.resource_monitor != "off":
            m.console.printf(
                "    measured cores: {:.1f}, memory: {:.0f} MB, disk {:.0f} MB, gpus: {:.1f}, runtime {:.1f} s",
                self.resources_measured.cores + 0.0,  # +0.0 trick to clear any -0.0
                self.resources_measured.memory,
                self.resources_measured.disk,
                self.resources_measured.gpus,
                (self.get_metric("time_workers_execute_last")) / 1e6,
            )

        if m.executor.print_stdout or not (self.successful() or self.exhausted()) or self.category == "accumulating":
            if self.std_output:
                m.console.print("    output:")
                m.console.print(self.std_output)

        if not (self.successful() or self.exhausted()):
            info = self.debug_info()
            m.console.warn(
                "task id {} item {} failed: {}\n    {}",
                self.id,
                self.itemid,
                self.result,
                info,
            )
        return self.successful()


class PreProcTask(CoffeaVineTask):
    def __init__(self, m, fn, item, itemid=None):
        if not itemid:
            itemid = "pre_{}".format(CoffeaVineTask.tasks_counter)

        self.item = item

        self.size = 1
        super().__init__(m, fn, [item], itemid, bring_back_output=True)

        self.set_category("preprocessing")
        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transfering file.
            pass
        else:
            f = m.declare_file(item.filename, cache=False)
            self.add_input(f, item.filename)

    def clone(self, m):
        return PreProcTask(
            m,
            self.function,
            self.item,
            self.itemid,
        )

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return "{} {}".format((i.dataset, i.filename, i.treename), msg)


class ProcTask(CoffeaVineTask):
    def __init__(self, m, fn, item, itemid=None, bring_back_output=False):
        self.size = len(item)

        if not itemid:
            itemid = "p_{}".format(CoffeaVineTask.tasks_counter)

        self.item = item

        super().__init__(m, fn, [item], itemid, bring_back_output=bring_back_output)

        self.set_category("processing")

        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transfering file.
            pass
        else:
            f = m.declare_file(item.filename, cache=False)
            self.add_input(f, item.filename)

    def clone(self, m):
        return ProcTask(
            m,
            self.function,
            self.item,
            self.itemid,
        )

    def resubmit(self, m):
        if self.retries_to_go < 1:
            return m.soft_terminate(self)

        if self.exhausted():
            if m.executor.split_on_exhaustion:
                return self.split(m)
            else:
                return m.soft_terminate(self)
        else:
            return super().resubmit(m)

    def split(self, m):
        m.console.warn(f"spliting task id {self.id} after resource exhaustion.")

        total = len(self.item)
        if total < 2:
            return m.soft_terminate()

        # if the chunksize was updated to be less than total, then use that.
        # Otherwise, partition the task in two and update the current chunksize.
        chunksize_target = m.chunksize_current
        if total <= chunksize_target:
            chunksize_target = math.ceil(total / 2)
            m.chunksize_current = chunksize_target

        n = max(math.ceil(total / chunksize_target), 1)
        chunksize_actual = int(math.ceil(total / n))

        m.stats_coffea["chunks_split"] += 1

        # remove the original item from the known work items, as it is being
        # split into two or more work items.
        m.known_workitems.remove(self.item)

        i = self.item
        start = i.entrystart
        while start < self.item.entrystop:
            stop = min(i.entrystop, start + chunksize_actual)
            w = WorkItem(
                i.dataset, i.filename, i.treename, start, stop, i.fileuuid, i.usermeta
            )
            t = self.__class__(m, self.function, w)
            start = stop

            m.submit(t)
            m.known_workitems.add(w)

            m.console(
                "resubmitting {} partly as {} with {} events. {} attempt(s) left.",
                self.itemid,
                t.itemid,
                len(t),
                t.retries_to_go,
            )

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return "{} {}".format(
            (i.dataset, i.filename, i.treename, i.entrystart, i.entrystop), msg
        )


class AccumTask(CoffeaVineTask):
    def __init__(
        self,
        m,
        fn,
        tasks_to_accumulate,
        itemid=None,
        bring_back_output=False
    ):
        if not itemid:
            itemid = "accum_{}".format(CoffeaVineTask.tasks_counter)

        self.tasks_to_accumulate = tasks_to_accumulate
        self.size = sum(len(t) for t in self.tasks_to_accumulate)

        names = [f"file.{i}" for (i, t) in enumerate(self.tasks_to_accumulate)]

        super().__init__(m, fn, [names], itemid, bring_back_output=bring_back_output)

        self.set_category("accumulating")
        for (name, t) in zip(names, self.tasks_to_accumulate):
            self.add_input(t.output_file, name)

    def cleanup_inputs(self, m):
        super().cleanup_inputs(m)
        # cleanup files associated with results already accumulated
        for t in self.tasks_to_accumulate:
            t.cleanup_outputs(m)

    def clone(self, m):
        return AccumTask(
            m,
            self.function,
            self.tasks_to_accumulate,
            self.itemid,
        )

    def debug_info(self):
        tasks = self.tasks_to_accumulate

        msg = super().debug_info()
        return "{} accumulating: [{}] ".format(msg, "\n".join([t.result for t in tasks]))


def run(executor, items, function, accumulator):
    """Execute using Work Queue
    For more information, see :ref:`intro-coffea-vine`
    """
    if not vine:
        print("You must have TaskVine installed to use TaskVineExecutor!")
        # raise an import error for taskvine
        import ndcctools.taskvine  # noqa

    global manager
    if manager is None:
        manager = CoffeaVine(executor)
    else:
        # if m already listening on port, update the parameters given by
        # the executor
        manager.executor = executor

    try:
        if executor.custom_init:
            executor.custom_init(manager)

        if executor.desc == "Preprocessing":
            result = manager._preprocessing(items, function, accumulator)
            # we do not shutdown m after preprocessing, as we want to
            # keep the connected workers for processing/accumulation
        else:
            result = manager._processing(items, function, accumulator)
            manager = None
    except Exception as e:
        manager = None
        raise e

    return result


def _handle_early_terminate(signum, frame, raise_on_repeat=True):
    global early_terminate

    if early_terminate and raise_on_repeat:
        raise KeyboardInterrupt
    else:
        manager.console.printf(
            "********************************************************************************"
        )
        manager.console.printf("Canceling processing tasks for final accumulation.")
        manager.console.printf("C-c now to immediately terminate.")
        manager.console.printf(
            "********************************************************************************"
        )
        early_terminate = True
        manager.cancel_by_category("processing")
        manager.cancel_by_category("accumulating")


def _get_x509_proxy(x509_proxy=None):
    if x509_proxy:
        return x509_proxy

    x509_proxy = os.environ.get("X509_USER_PROXY", None)
    if x509_proxy:
        return x509_proxy

    x509_proxy = join(
        os.environ.get("TMPDIR", "/tmp"), "x509up_u{}".format(os.getuid())
    )
    if os.path.exists(x509_proxy):
        return x509_proxy

    return None


class ResultUnavailable(Exception):
    pass


class Stats(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(int, *args, **kwargs)

    def min(self, stat, value):
        try:
            self[stat] = min(self[stat], value)
        except KeyError:
            self[stat] = value

    def max(self, stat, value):
        try:
            self[stat] = max(self[stat], value)
        except KeyError:
            self[stat] = value


class VerbosePrint:
    def __init__(self, console, status_mode=True, verbose_mode=True):
        self.console = console
        self.status_mode = status_mode
        self.verbose_mode = verbose_mode

    def __call__(self, format_str, *args, **kwargs):
        if self.verbose_mode:
            self.printf(format_str, *args, **kwargs)

    def print(self, msg):
        if self.status_mode:
            self.console.print(msg)
        else:
            print(msg)

    def printf(self, format_str, *args, **kwargs):
        msg = format_str.format(*args, **kwargs)
        self.print(msg)

    def warn(self, format_str, *args, **kwargs):
        if self.status_mode:
            format_str = "[red]WARNING:[/red] " + format_str
        else:
            format_str = "WARNING: " + format_str
        self.printf(format_str, *args, **kwargs)


# Support for rich_bar so that we can keep track of bars by their names, rather
# than the changing bar ids.
class StatusBar:
    def __init__(self, enabled=True):
        self._prog = rich_bar()
        self._ids = {}
        if enabled:
            self._prog.start()

    def add_task(self, desc, *args, **kwargs):
        b = self._prog.add_task(desc, *args, **kwargs)
        self._ids[desc] = b
        self._prog.start_task(self._ids[desc])
        return b

    def stop_task(self, desc, *args, **kwargs):
        return self._prog.stop_task(self._ids[desc], *args, **kwargs)

    def update(self, desc, *args, **kwargs):
        return self._prog.update(self._ids[desc], *args, **kwargs)

    def advance(self, desc, *args, **kwargs):
        return self._prog.advance(self._ids[desc], *args, **kwargs)

    # redirect anything else to rich_bar
    def __getattr__(self, name):
        return getattr(self._prog, name)
