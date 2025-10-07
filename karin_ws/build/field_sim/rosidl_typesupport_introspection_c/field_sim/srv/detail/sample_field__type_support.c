// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "field_sim/srv/detail/sample_field__rosidl_typesupport_introspection_c.h"
#include "field_sim/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "field_sim/srv/detail/sample_field__functions.h"
#include "field_sim/srv/detail/sample_field__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  field_sim__srv__SampleField_Request__init(message_memory);
}

void field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_fini_function(void * message_memory)
{
  field_sim__srv__SampleField_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_member_array[2] = {
  {
    "latitude",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Request, latitude),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "longitude",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Request, longitude),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_members = {
  "field_sim__srv",  // message namespace
  "SampleField_Request",  // message name
  2,  // number of fields
  sizeof(field_sim__srv__SampleField_Request),
  false,  // has_any_key_member_
  field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_member_array,  // message members
  field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_type_support_handle = {
  0,
  &field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_members,
  get_message_typesupport_handle_function,
  &field_sim__srv__SampleField_Request__get_type_hash,
  &field_sim__srv__SampleField_Request__get_type_description,
  &field_sim__srv__SampleField_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_field_sim
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Request)() {
  if (!field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_type_support_handle.typesupport_identifier) {
    field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "field_sim/srv/detail/sample_field__rosidl_typesupport_introspection_c.h"
// already included above
// #include "field_sim/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "field_sim/srv/detail/sample_field__functions.h"
// already included above
// #include "field_sim/srv/detail/sample_field__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  field_sim__srv__SampleField_Response__init(message_memory);
}

void field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_fini_function(void * message_memory)
{
  field_sim__srv__SampleField_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_member_array[4] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "temperature",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Response, temperature),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x_enu",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Response, x_enu),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y_enu",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Response, y_enu),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_members = {
  "field_sim__srv",  // message namespace
  "SampleField_Response",  // message name
  4,  // number of fields
  sizeof(field_sim__srv__SampleField_Response),
  false,  // has_any_key_member_
  field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_member_array,  // message members
  field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle = {
  0,
  &field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_members,
  get_message_typesupport_handle_function,
  &field_sim__srv__SampleField_Response__get_type_hash,
  &field_sim__srv__SampleField_Response__get_type_description,
  &field_sim__srv__SampleField_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_field_sim
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Response)() {
  if (!field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle.typesupport_identifier) {
    field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "field_sim/srv/detail/sample_field__rosidl_typesupport_introspection_c.h"
// already included above
// #include "field_sim/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "field_sim/srv/detail/sample_field__functions.h"
// already included above
// #include "field_sim/srv/detail/sample_field__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "field_sim/srv/sample_field.h"
// Member `request`
// Member `response`
// already included above
// #include "field_sim/srv/detail/sample_field__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  field_sim__srv__SampleField_Event__init(message_memory);
}

void field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_fini_function(void * message_memory)
{
  field_sim__srv__SampleField_Event__fini(message_memory);
}

size_t field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__size_function__SampleField_Event__request(
  const void * untyped_member)
{
  const field_sim__srv__SampleField_Request__Sequence * member =
    (const field_sim__srv__SampleField_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_const_function__SampleField_Event__request(
  const void * untyped_member, size_t index)
{
  const field_sim__srv__SampleField_Request__Sequence * member =
    (const field_sim__srv__SampleField_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_function__SampleField_Event__request(
  void * untyped_member, size_t index)
{
  field_sim__srv__SampleField_Request__Sequence * member =
    (field_sim__srv__SampleField_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__fetch_function__SampleField_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const field_sim__srv__SampleField_Request * item =
    ((const field_sim__srv__SampleField_Request *)
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_const_function__SampleField_Event__request(untyped_member, index));
  field_sim__srv__SampleField_Request * value =
    (field_sim__srv__SampleField_Request *)(untyped_value);
  *value = *item;
}

void field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__assign_function__SampleField_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  field_sim__srv__SampleField_Request * item =
    ((field_sim__srv__SampleField_Request *)
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_function__SampleField_Event__request(untyped_member, index));
  const field_sim__srv__SampleField_Request * value =
    (const field_sim__srv__SampleField_Request *)(untyped_value);
  *item = *value;
}

bool field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__resize_function__SampleField_Event__request(
  void * untyped_member, size_t size)
{
  field_sim__srv__SampleField_Request__Sequence * member =
    (field_sim__srv__SampleField_Request__Sequence *)(untyped_member);
  field_sim__srv__SampleField_Request__Sequence__fini(member);
  return field_sim__srv__SampleField_Request__Sequence__init(member, size);
}

size_t field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__size_function__SampleField_Event__response(
  const void * untyped_member)
{
  const field_sim__srv__SampleField_Response__Sequence * member =
    (const field_sim__srv__SampleField_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_const_function__SampleField_Event__response(
  const void * untyped_member, size_t index)
{
  const field_sim__srv__SampleField_Response__Sequence * member =
    (const field_sim__srv__SampleField_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_function__SampleField_Event__response(
  void * untyped_member, size_t index)
{
  field_sim__srv__SampleField_Response__Sequence * member =
    (field_sim__srv__SampleField_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__fetch_function__SampleField_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const field_sim__srv__SampleField_Response * item =
    ((const field_sim__srv__SampleField_Response *)
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_const_function__SampleField_Event__response(untyped_member, index));
  field_sim__srv__SampleField_Response * value =
    (field_sim__srv__SampleField_Response *)(untyped_value);
  *value = *item;
}

void field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__assign_function__SampleField_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  field_sim__srv__SampleField_Response * item =
    ((field_sim__srv__SampleField_Response *)
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_function__SampleField_Event__response(untyped_member, index));
  const field_sim__srv__SampleField_Response * value =
    (const field_sim__srv__SampleField_Response *)(untyped_value);
  *item = *value;
}

bool field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__resize_function__SampleField_Event__response(
  void * untyped_member, size_t size)
{
  field_sim__srv__SampleField_Response__Sequence * member =
    (field_sim__srv__SampleField_Response__Sequence *)(untyped_member);
  field_sim__srv__SampleField_Response__Sequence__fini(member);
  return field_sim__srv__SampleField_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(field_sim__srv__SampleField_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(field_sim__srv__SampleField_Event, request),  // bytes offset in struct
    NULL,  // default value
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__size_function__SampleField_Event__request,  // size() function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_const_function__SampleField_Event__request,  // get_const(index) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_function__SampleField_Event__request,  // get(index) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__fetch_function__SampleField_Event__request,  // fetch(index, &value) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__assign_function__SampleField_Event__request,  // assign(index, value) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__resize_function__SampleField_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(field_sim__srv__SampleField_Event, response),  // bytes offset in struct
    NULL,  // default value
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__size_function__SampleField_Event__response,  // size() function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_const_function__SampleField_Event__response,  // get_const(index) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__get_function__SampleField_Event__response,  // get(index) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__fetch_function__SampleField_Event__response,  // fetch(index, &value) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__assign_function__SampleField_Event__response,  // assign(index, value) function pointer
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__resize_function__SampleField_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_members = {
  "field_sim__srv",  // message namespace
  "SampleField_Event",  // message name
  3,  // number of fields
  sizeof(field_sim__srv__SampleField_Event),
  false,  // has_any_key_member_
  field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_member_array,  // message members
  field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_type_support_handle = {
  0,
  &field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_members,
  get_message_typesupport_handle_function,
  &field_sim__srv__SampleField_Event__get_type_hash,
  &field_sim__srv__SampleField_Event__get_type_description,
  &field_sim__srv__SampleField_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_field_sim
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Event)() {
  field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Request)();
  field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Response)();
  if (!field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_type_support_handle.typesupport_identifier) {
    field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "field_sim/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "field_sim/srv/detail/sample_field__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_members = {
  "field_sim__srv",  // service namespace
  "SampleField",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_Request_message_type_support_handle,
  NULL,  // response message
  // field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle
  NULL  // event_message
  // field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle
};


static rosidl_service_type_support_t field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_type_support_handle = {
  0,
  &field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_members,
  get_service_typesupport_handle_function,
  &field_sim__srv__SampleField_Request__rosidl_typesupport_introspection_c__SampleField_Request_message_type_support_handle,
  &field_sim__srv__SampleField_Response__rosidl_typesupport_introspection_c__SampleField_Response_message_type_support_handle,
  &field_sim__srv__SampleField_Event__rosidl_typesupport_introspection_c__SampleField_Event_message_type_support_handle,
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

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_field_sim
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField)(void) {
  if (!field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_type_support_handle.typesupport_identifier) {
    field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, field_sim, srv, SampleField_Event)()->data;
  }

  return &field_sim__srv__detail__sample_field__rosidl_typesupport_introspection_c__SampleField_service_type_support_handle;
}
