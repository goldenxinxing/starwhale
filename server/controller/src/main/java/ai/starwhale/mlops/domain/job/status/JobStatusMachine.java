/*
 * Copyright 2022 Starwhale, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.starwhale.mlops.domain.job.status;


import static ai.starwhale.mlops.domain.job.status.JobStatus.CANCELED;
import static ai.starwhale.mlops.domain.job.status.JobStatus.CANCELLING;
import static ai.starwhale.mlops.domain.job.status.JobStatus.CREATED;
import static ai.starwhale.mlops.domain.job.status.JobStatus.FAIL;
import static ai.starwhale.mlops.domain.job.status.JobStatus.PAUSED;
import static ai.starwhale.mlops.domain.job.status.JobStatus.READY;
import static ai.starwhale.mlops.domain.job.status.JobStatus.RUNNING;
import static ai.starwhale.mlops.domain.job.status.JobStatus.SUCCESS;
import static ai.starwhale.mlops.domain.job.status.JobStatus.UNKNOWN;

import java.util.Map;
import java.util.Set;

public class JobStatusMachine {

    static final Map<JobStatus, Set<JobStatus>> transferMap = Map.of(
            CREATED, Set.of(READY, PAUSED, RUNNING, SUCCESS, CANCELLING, FAIL),
            READY, Set.of(PAUSED, RUNNING, SUCCESS, CANCELLING, FAIL),
            PAUSED, Set.of(READY, RUNNING, CANCELED, FAIL),
            RUNNING, Set.of(PAUSED, SUCCESS, CANCELLING, FAIL),
            CANCELLING, Set.of(CANCELED, FAIL),
            SUCCESS, Set.of(),
            FAIL, Set.of(),
            CANCELED, Set.of(),
            UNKNOWN, Set.of(JobStatus.values()));

    public static final Set<JobStatus> HOT_JOB_STATUS = Set.of(READY, RUNNING, CANCELLING);
    public static final Set<JobStatus> FINAL_STATUS = Set.of(FAIL, SUCCESS, CANCELED);

    public static boolean couldTransfer(JobStatus statusNow, JobStatus statusNew) {
        return transferMap.get(statusNow).contains(statusNew);
    }

    public static boolean isHot(JobStatus status) {
        return HOT_JOB_STATUS.contains(status);
    }

    public static boolean isFinal(JobStatus status) {
        return FINAL_STATUS.contains(status);
    }

}

