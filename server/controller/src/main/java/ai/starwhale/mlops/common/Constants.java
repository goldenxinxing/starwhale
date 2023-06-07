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

package ai.starwhale.mlops.common;

import com.fasterxml.jackson.dataformat.yaml.YAMLMapper;
import org.codehaus.jackson.map.ObjectMapper;

public interface Constants {
    YAMLMapper yamlMapper = new YAMLMapper();
    String SW_BUILT_IN_RUNTIME = "starwhale-built-in";
    ObjectMapper objectMapper = new ObjectMapper();
}
