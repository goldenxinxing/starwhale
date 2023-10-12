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

package ai.starwhale.mlops.domain.dataset.dataloader.mapper;

import ai.starwhale.mlops.domain.dataset.dataloader.po.DataReadLogEntity;
import java.util.List;
import java.util.Objects;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;
import org.apache.ibatis.annotations.UpdateProvider;
import org.apache.ibatis.jdbc.SQL;

@Mapper
public interface DataReadLogMapper {

    @Insert({
        "<script>",
        "INSERT INTO dataset_read_log"
            + "(id, session_id, start, start_type, start_inclusive, end, end_type, end_inclusive, size, status)",
        "VALUES"
            + "<foreach item='data' collection='records' open='' separator=',' close=''>"
            + "("
            + "#{data.id}, #{data.sessionId}, #{data.start}, #{data.startType}, #{data.startInclusive},"
            + "#{data.end}, #{data.endType}, #{data.endInclusive}, #{data.size}, #{data.status}"
            + ")"
            + "</foreach>",
        "</script>"})
    int batchInsert(@Param("records") List<DataReadLogEntity> theCollection);

    @Update("UPDATE dataset_read_log SET "
            + "consumer_id=#{consumerId}, "
            + "status=#{status}, assigned_time=CURRENT_TIMESTAMP(), assigned_num=assigned_num+1 "
            + "WHERE id=#{id}")
    int updateToAssigned(DataReadLogEntity dataBlock);

    class UpdateToProcessedSqlProvider {
        public static String updateToProcessedSql(Long sessionId,
                                                  String consumerId,
                                                  String start,
                                                  String end,
                                                  String status) {
            return new SQL() {{
                    UPDATE("dataset_read_log");
                    SET("consumer_id=#{consumerId}", "status=#{status}", "finished_time=NOW()");
                    WHERE("session_id=#{sessionId}", "start=#{start}");
                    if (Objects.isNull(end)) {
                        WHERE("end is null");
                    } else {
                        WHERE("end=#{end}");
                    }
                }}
                .toString();
        }
    }

    @UpdateProvider(value = UpdateToProcessedSqlProvider.class, method = "updateToProcessedSql")
    int updateToProcessed(@Param("sessionId") Long sessionId, @Param("consumerId") String consumerId,
                          @Param("start") String start, @Param("end") String end, @Param("status") String status);

    @Update("UPDATE dataset_read_log SET "
            + "consumer_id=null "
            + "WHERE session_id=#{sessionId} and consumer_id=#{consumerId} and status=#{status}")
    int updateToUnAssigned(
            @Param("sessionId") Long sessionId, @Param("consumerId") String consumerId, @Param("status") String status);

    @Update("UPDATE dataset_read_log SET "
            + "consumer_id=null "
            + "WHERE consumer_id=#{consumerId} and status=#{status}")
    int updateToUnAssignedForConsumer(@Param("consumerId") String consumerId, @Param("status") String status);

    @Select("SELECT * from dataset_read_log "
            + "WHERE session_id=#{sessionId} and (consumer_id is null or consumer_id = '') and status=#{status} "
            + "ORDER BY id "
            + "LIMIT #{limit}")
    List<DataReadLogEntity> selectTopsUnAssigned(
            @Param("sessionId") Long sessionId, @Param("status") String status, @Param("limit") Integer limit);

    @Select("SELECT * from dataset_read_log "
            + "WHERE session_id in "
            + "(SELECT id from dataset_read_session where session_id=#{sessionId}) and status=#{status} ")
    List<DataReadLogEntity> selectByStatus(@Param("sessionId") String sessionId, @Param("status") String status);

    @Select("SELECT * from dataset_read_log WHERE id=#{id} ")
    DataReadLogEntity selectOne(@Param("id") Long id);

    @Select("SELECT sum(assigned_num) from dataset_read_log "
            + "WHERE session_id in (SELECT id from dataset_read_session where session_id=#{sessionId})")
    int totalAssignedNum(@Param("sessionId") String sessionId);
}
