// generated from rosidl_generator_c/resource/idl__type_support.c.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

#include <string.h>

#include "field_sim/srv/detail/sample_field__struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "field_sim/srv/detail/sample_field__functions.h"
#include "field_sim/srv/detail/sample_field__type_support.h"

#ifdef __cplusplus
extern "C"
{
#endif


void *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
  rosidl_typesupport_c,
  field_sim,
  srv,
  SampleField
)(
  const rosidl_service_introspection_info_t * info,
  rcutils_allocator_t * allocator,
  const void * request_message,
  const void * response_message)
{
  if (!allocator || !info) {
    return NULL;
  }
  field_sim__srv__SampleField_Event * event_msg = (field_sim__srv__SampleField_Event *)(allocator->allocate(sizeof(field_sim__srv__SampleField_Event), allocator->state));
  if (!field_sim__srv__SampleField_Event__init(event_msg)) {
    allocator->deallocate(event_msg, allocator->state);
    return NULL;
  }

  event_msg->info.event_type = info->event_type;
  event_msg->info.sequence_number = info->sequence_number;
  event_msg->info.stamp.sec = info->stamp_sec;
  event_msg->info.stamp.nanosec = info->stamp_nanosec;
  memcpy(event_msg->info.client_gid, info->client_gid, 16);
  if (request_message) {
    field_sim__srv__SampleField_Request__Sequence__init(
      &event_msg->request,
      1);
    if (!field_sim__srv__SampleField_Request__copy((const field_sim__srv__SampleField_Request *)(request_message), event_msg->request.data)) {
      allocator->deallocate(event_msg, allocator->state);
      return NULL;
    }
  }
  if (response_message) {
    field_sim__srv__SampleField_Response__Sequence__init(
      &event_msg->response,
      1);
    if (!field_sim__srv__SampleField_Response__copy((const field_sim__srv__SampleField_Response *)(response_message), event_msg->response.data)) {
      allocator->deallocate(event_msg, allocator->state);
      return NULL;
    }
  }
  return event_msg;
}

// Forward declare the get type support functions for this type.
bool
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
  rosidl_typesupport_c,
  field_sim,
  srv,
  SampleField
)(
  void * event_msg,
  rcutils_allocator_t * allocator)
{
  if (!allocator) {
    return false;
  }
  if (NULL == event_msg) {
    return false;
  }
  field_sim__srv__SampleField_Event * _event_msg = (field_sim__srv__SampleField_Event *)(event_msg);

  field_sim__srv__SampleField_Event__fini((field_sim__srv__SampleField_Event *)(_event_msg));
  if (_event_msg->request.data) {
    allocator->deallocate(_event_msg->request.data, allocator->state);
  }
  if (_event_msg->response.data) {
    allocator->deallocate(_event_msg->response.data, allocator->state);
  }
  allocator->deallocate(_event_msg, allocator->state);
  return true;
}

#ifdef __cplusplus
}
#endif
