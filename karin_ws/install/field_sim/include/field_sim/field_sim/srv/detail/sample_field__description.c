// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

#include "field_sim/srv/detail/sample_field__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_field_sim
const rosidl_type_hash_t *
field_sim__srv__SampleField__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x68, 0x6b, 0x15, 0x63, 0x63, 0x6a, 0x31, 0xfc,
      0xb3, 0xdb, 0x47, 0x42, 0x4d, 0x3a, 0x25, 0xc8,
      0xba, 0xf6, 0x57, 0xd1, 0x0e, 0x61, 0x6a, 0xf2,
      0x4f, 0xff, 0x28, 0xb7, 0x4a, 0xe9, 0xb6, 0xbd,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_field_sim
const rosidl_type_hash_t *
field_sim__srv__SampleField_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x3b, 0x14, 0x86, 0xcf, 0xd0, 0x64, 0x58, 0xb3,
      0x3f, 0x03, 0x54, 0xf3, 0x28, 0x73, 0x49, 0x7e,
      0x3f, 0x43, 0xfd, 0xb6, 0x91, 0xf5, 0x5a, 0x81,
      0x4e, 0x0e, 0x26, 0x22, 0x7c, 0x73, 0x0d, 0x2b,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_field_sim
const rosidl_type_hash_t *
field_sim__srv__SampleField_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x0e, 0x64, 0xeb, 0xd5, 0x6b, 0x19, 0x51, 0xd7,
      0xaa, 0x4d, 0xd8, 0x0f, 0xbc, 0x41, 0xcd, 0x6f,
      0x2f, 0x92, 0x66, 0xfd, 0x13, 0x3c, 0x9f, 0xb3,
      0x9b, 0x7c, 0xd3, 0xf8, 0x65, 0xcf, 0x6d, 0x5c,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_field_sim
const rosidl_type_hash_t *
field_sim__srv__SampleField_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xf4, 0x26, 0x46, 0x3e, 0x40, 0x09, 0x73, 0x27,
      0x4b, 0x3c, 0x9d, 0xc3, 0xcc, 0x66, 0x88, 0x28,
      0x3f, 0xd6, 0x71, 0xef, 0x6f, 0xee, 0xdc, 0xbb,
      0x79, 0x13, 0x26, 0xe7, 0x62, 0xa3, 0xa3, 0xe7,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char field_sim__srv__SampleField__TYPE_NAME[] = "field_sim/srv/SampleField";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char field_sim__srv__SampleField_Event__TYPE_NAME[] = "field_sim/srv/SampleField_Event";
static char field_sim__srv__SampleField_Request__TYPE_NAME[] = "field_sim/srv/SampleField_Request";
static char field_sim__srv__SampleField_Response__TYPE_NAME[] = "field_sim/srv/SampleField_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char field_sim__srv__SampleField__FIELD_NAME__request_message[] = "request_message";
static char field_sim__srv__SampleField__FIELD_NAME__response_message[] = "response_message";
static char field_sim__srv__SampleField__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field field_sim__srv__SampleField__FIELDS[] = {
  {
    {field_sim__srv__SampleField__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {field_sim__srv__SampleField_Request__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {field_sim__srv__SampleField_Response__TYPE_NAME, 34, 34},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {field_sim__srv__SampleField_Event__TYPE_NAME, 31, 31},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription field_sim__srv__SampleField__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Event__TYPE_NAME, 31, 31},
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Request__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Response__TYPE_NAME, 34, 34},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
field_sim__srv__SampleField__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {field_sim__srv__SampleField__TYPE_NAME, 25, 25},
      {field_sim__srv__SampleField__FIELDS, 3, 3},
    },
    {field_sim__srv__SampleField__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = field_sim__srv__SampleField_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = field_sim__srv__SampleField_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = field_sim__srv__SampleField_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char field_sim__srv__SampleField_Request__FIELD_NAME__latitude[] = "latitude";
static char field_sim__srv__SampleField_Request__FIELD_NAME__longitude[] = "longitude";

static rosidl_runtime_c__type_description__Field field_sim__srv__SampleField_Request__FIELDS[] = {
  {
    {field_sim__srv__SampleField_Request__FIELD_NAME__latitude, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Request__FIELD_NAME__longitude, 9, 9},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
field_sim__srv__SampleField_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {field_sim__srv__SampleField_Request__TYPE_NAME, 33, 33},
      {field_sim__srv__SampleField_Request__FIELDS, 2, 2},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char field_sim__srv__SampleField_Response__FIELD_NAME__success[] = "success";
static char field_sim__srv__SampleField_Response__FIELD_NAME__temperature[] = "temperature";
static char field_sim__srv__SampleField_Response__FIELD_NAME__x_enu[] = "x_enu";
static char field_sim__srv__SampleField_Response__FIELD_NAME__y_enu[] = "y_enu";

static rosidl_runtime_c__type_description__Field field_sim__srv__SampleField_Response__FIELDS[] = {
  {
    {field_sim__srv__SampleField_Response__FIELD_NAME__success, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_BOOLEAN,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Response__FIELD_NAME__temperature, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Response__FIELD_NAME__x_enu, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Response__FIELD_NAME__y_enu, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_DOUBLE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
field_sim__srv__SampleField_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {field_sim__srv__SampleField_Response__TYPE_NAME, 34, 34},
      {field_sim__srv__SampleField_Response__FIELDS, 4, 4},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char field_sim__srv__SampleField_Event__FIELD_NAME__info[] = "info";
static char field_sim__srv__SampleField_Event__FIELD_NAME__request[] = "request";
static char field_sim__srv__SampleField_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field field_sim__srv__SampleField_Event__FIELDS[] = {
  {
    {field_sim__srv__SampleField_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {field_sim__srv__SampleField_Request__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {field_sim__srv__SampleField_Response__TYPE_NAME, 34, 34},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription field_sim__srv__SampleField_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Request__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {field_sim__srv__SampleField_Response__TYPE_NAME, 34, 34},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
field_sim__srv__SampleField_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {field_sim__srv__SampleField_Event__TYPE_NAME, 31, 31},
      {field_sim__srv__SampleField_Event__FIELDS, 3, 3},
    },
    {field_sim__srv__SampleField_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = field_sim__srv__SampleField_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = field_sim__srv__SampleField_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "float64 latitude\n"
  "float64 longitude\n"
  "---\n"
  "bool success\n"
  "float64 temperature\n"
  "float64 x_enu\n"
  "float64 y_enu";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
field_sim__srv__SampleField__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {field_sim__srv__SampleField__TYPE_NAME, 25, 25},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 99, 99},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
field_sim__srv__SampleField_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {field_sim__srv__SampleField_Request__TYPE_NAME, 33, 33},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
field_sim__srv__SampleField_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {field_sim__srv__SampleField_Response__TYPE_NAME, 34, 34},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
field_sim__srv__SampleField_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {field_sim__srv__SampleField_Event__TYPE_NAME, 31, 31},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
field_sim__srv__SampleField__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *field_sim__srv__SampleField__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *field_sim__srv__SampleField_Event__get_individual_type_description_source(NULL);
    sources[3] = *field_sim__srv__SampleField_Request__get_individual_type_description_source(NULL);
    sources[4] = *field_sim__srv__SampleField_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
field_sim__srv__SampleField_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *field_sim__srv__SampleField_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
field_sim__srv__SampleField_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *field_sim__srv__SampleField_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
field_sim__srv__SampleField_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *field_sim__srv__SampleField_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *field_sim__srv__SampleField_Request__get_individual_type_description_source(NULL);
    sources[3] = *field_sim__srv__SampleField_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
