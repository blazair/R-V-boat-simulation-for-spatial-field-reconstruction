// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "field_sim/srv/detail/sample_field__struct.h"
#include "field_sim/srv/detail/sample_field__type_support.h"
#include "field_sim/srv/detail/sample_field__functions.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace field_sim
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _SampleField_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SampleField_Request_type_support_ids_t;

static const _SampleField_Request_type_support_ids_t _SampleField_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SampleField_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SampleField_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SampleField_Request_type_support_symbol_names_t _SampleField_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, field_sim, srv, SampleField_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Request)),
  }
};

typedef struct _SampleField_Request_type_support_data_t
{
  void * data[2];
} _SampleField_Request_type_support_data_t;

static _SampleField_Request_type_support_data_t _SampleField_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SampleField_Request_message_typesupport_map = {
  2,
  "field_sim",
  &_SampleField_Request_message_typesupport_ids.typesupport_identifier[0],
  &_SampleField_Request_message_typesupport_symbol_names.symbol_name[0],
  &_SampleField_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t SampleField_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SampleField_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &field_sim__srv__SampleField_Request__get_type_hash,
  &field_sim__srv__SampleField_Request__get_type_description,
  &field_sim__srv__SampleField_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace field_sim

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, field_sim, srv, SampleField_Request)() {
  return &::field_sim::srv::rosidl_typesupport_c::SampleField_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "field_sim/srv/detail/sample_field__struct.h"
// already included above
// #include "field_sim/srv/detail/sample_field__type_support.h"
// already included above
// #include "field_sim/srv/detail/sample_field__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace field_sim
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _SampleField_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SampleField_Response_type_support_ids_t;

static const _SampleField_Response_type_support_ids_t _SampleField_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SampleField_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SampleField_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SampleField_Response_type_support_symbol_names_t _SampleField_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, field_sim, srv, SampleField_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Response)),
  }
};

typedef struct _SampleField_Response_type_support_data_t
{
  void * data[2];
} _SampleField_Response_type_support_data_t;

static _SampleField_Response_type_support_data_t _SampleField_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SampleField_Response_message_typesupport_map = {
  2,
  "field_sim",
  &_SampleField_Response_message_typesupport_ids.typesupport_identifier[0],
  &_SampleField_Response_message_typesupport_symbol_names.symbol_name[0],
  &_SampleField_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t SampleField_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SampleField_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &field_sim__srv__SampleField_Response__get_type_hash,
  &field_sim__srv__SampleField_Response__get_type_description,
  &field_sim__srv__SampleField_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace field_sim

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, field_sim, srv, SampleField_Response)() {
  return &::field_sim::srv::rosidl_typesupport_c::SampleField_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "field_sim/srv/detail/sample_field__struct.h"
// already included above
// #include "field_sim/srv/detail/sample_field__type_support.h"
// already included above
// #include "field_sim/srv/detail/sample_field__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace field_sim
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _SampleField_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SampleField_Event_type_support_ids_t;

static const _SampleField_Event_type_support_ids_t _SampleField_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SampleField_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SampleField_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SampleField_Event_type_support_symbol_names_t _SampleField_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, field_sim, srv, SampleField_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Event)),
  }
};

typedef struct _SampleField_Event_type_support_data_t
{
  void * data[2];
} _SampleField_Event_type_support_data_t;

static _SampleField_Event_type_support_data_t _SampleField_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SampleField_Event_message_typesupport_map = {
  2,
  "field_sim",
  &_SampleField_Event_message_typesupport_ids.typesupport_identifier[0],
  &_SampleField_Event_message_typesupport_symbol_names.symbol_name[0],
  &_SampleField_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t SampleField_Event_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SampleField_Event_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &field_sim__srv__SampleField_Event__get_type_hash,
  &field_sim__srv__SampleField_Event__get_type_description,
  &field_sim__srv__SampleField_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace field_sim

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, field_sim, srv, SampleField_Event)() {
  return &::field_sim::srv::rosidl_typesupport_c::SampleField_Event_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "field_sim/srv/detail/sample_field__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
#include "service_msgs/msg/service_event_info.h"
#include "builtin_interfaces/msg/time.h"

namespace field_sim
{

namespace srv
{

namespace rosidl_typesupport_c
{
typedef struct _SampleField_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SampleField_type_support_ids_t;

static const _SampleField_type_support_ids_t _SampleField_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SampleField_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SampleField_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SampleField_type_support_symbol_names_t _SampleField_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, field_sim, srv, SampleField)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField)),
  }
};

typedef struct _SampleField_type_support_data_t
{
  void * data[2];
} _SampleField_type_support_data_t;

static _SampleField_type_support_data_t _SampleField_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SampleField_service_typesupport_map = {
  2,
  "field_sim",
  &_SampleField_service_typesupport_ids.typesupport_identifier[0],
  &_SampleField_service_typesupport_symbol_names.symbol_name[0],
  &_SampleField_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t SampleField_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SampleField_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
  &SampleField_Request_message_type_support_handle,
  &SampleField_Response_message_type_support_handle,
  &SampleField_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    field_sim,
    srv,
    SampleField
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    field_sim,
    srv,
    SampleField
  ),
  &field_sim__srv__SampleField__get_type_hash,
  &field_sim__srv__SampleField__get_type_description,
  &field_sim__srv__SampleField__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace field_sim

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, field_sim, srv, SampleField)() {
  return &::field_sim::srv::rosidl_typesupport_c::SampleField_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif
