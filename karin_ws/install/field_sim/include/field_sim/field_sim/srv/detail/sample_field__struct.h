// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "field_sim/srv/sample_field.h"


#ifndef FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__STRUCT_H_
#define FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SampleField in the package field_sim.
typedef struct field_sim__srv__SampleField_Request
{
  double latitude;
  double longitude;
} field_sim__srv__SampleField_Request;

// Struct for a sequence of field_sim__srv__SampleField_Request.
typedef struct field_sim__srv__SampleField_Request__Sequence
{
  field_sim__srv__SampleField_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} field_sim__srv__SampleField_Request__Sequence;

// Constants defined in the message

/// Struct defined in srv/SampleField in the package field_sim.
typedef struct field_sim__srv__SampleField_Response
{
  bool success;
  double temperature;
  double x_enu;
  double y_enu;
} field_sim__srv__SampleField_Response;

// Struct for a sequence of field_sim__srv__SampleField_Response.
typedef struct field_sim__srv__SampleField_Response__Sequence
{
  field_sim__srv__SampleField_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} field_sim__srv__SampleField_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  field_sim__srv__SampleField_Event__request__MAX_SIZE = 1
};
// response
enum
{
  field_sim__srv__SampleField_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/SampleField in the package field_sim.
typedef struct field_sim__srv__SampleField_Event
{
  service_msgs__msg__ServiceEventInfo info;
  field_sim__srv__SampleField_Request__Sequence request;
  field_sim__srv__SampleField_Response__Sequence response;
} field_sim__srv__SampleField_Event;

// Struct for a sequence of field_sim__srv__SampleField_Event.
typedef struct field_sim__srv__SampleField_Event__Sequence
{
  field_sim__srv__SampleField_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} field_sim__srv__SampleField_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__STRUCT_H_
