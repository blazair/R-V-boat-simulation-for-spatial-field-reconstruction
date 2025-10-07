// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice
#include "field_sim/srv/detail/sample_field__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
field_sim__srv__SampleField_Request__init(field_sim__srv__SampleField_Request * msg)
{
  if (!msg) {
    return false;
  }
  // latitude
  // longitude
  return true;
}

void
field_sim__srv__SampleField_Request__fini(field_sim__srv__SampleField_Request * msg)
{
  if (!msg) {
    return;
  }
  // latitude
  // longitude
}

bool
field_sim__srv__SampleField_Request__are_equal(const field_sim__srv__SampleField_Request * lhs, const field_sim__srv__SampleField_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // latitude
  if (lhs->latitude != rhs->latitude) {
    return false;
  }
  // longitude
  if (lhs->longitude != rhs->longitude) {
    return false;
  }
  return true;
}

bool
field_sim__srv__SampleField_Request__copy(
  const field_sim__srv__SampleField_Request * input,
  field_sim__srv__SampleField_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // latitude
  output->latitude = input->latitude;
  // longitude
  output->longitude = input->longitude;
  return true;
}

field_sim__srv__SampleField_Request *
field_sim__srv__SampleField_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Request * msg = (field_sim__srv__SampleField_Request *)allocator.allocate(sizeof(field_sim__srv__SampleField_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(field_sim__srv__SampleField_Request));
  bool success = field_sim__srv__SampleField_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
field_sim__srv__SampleField_Request__destroy(field_sim__srv__SampleField_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    field_sim__srv__SampleField_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
field_sim__srv__SampleField_Request__Sequence__init(field_sim__srv__SampleField_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Request * data = NULL;

  if (size) {
    data = (field_sim__srv__SampleField_Request *)allocator.zero_allocate(size, sizeof(field_sim__srv__SampleField_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = field_sim__srv__SampleField_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        field_sim__srv__SampleField_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
field_sim__srv__SampleField_Request__Sequence__fini(field_sim__srv__SampleField_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      field_sim__srv__SampleField_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

field_sim__srv__SampleField_Request__Sequence *
field_sim__srv__SampleField_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Request__Sequence * array = (field_sim__srv__SampleField_Request__Sequence *)allocator.allocate(sizeof(field_sim__srv__SampleField_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = field_sim__srv__SampleField_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
field_sim__srv__SampleField_Request__Sequence__destroy(field_sim__srv__SampleField_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    field_sim__srv__SampleField_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
field_sim__srv__SampleField_Request__Sequence__are_equal(const field_sim__srv__SampleField_Request__Sequence * lhs, const field_sim__srv__SampleField_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!field_sim__srv__SampleField_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
field_sim__srv__SampleField_Request__Sequence__copy(
  const field_sim__srv__SampleField_Request__Sequence * input,
  field_sim__srv__SampleField_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(field_sim__srv__SampleField_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    field_sim__srv__SampleField_Request * data =
      (field_sim__srv__SampleField_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!field_sim__srv__SampleField_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          field_sim__srv__SampleField_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!field_sim__srv__SampleField_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
field_sim__srv__SampleField_Response__init(field_sim__srv__SampleField_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // temperature
  // x_enu
  // y_enu
  return true;
}

void
field_sim__srv__SampleField_Response__fini(field_sim__srv__SampleField_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // temperature
  // x_enu
  // y_enu
}

bool
field_sim__srv__SampleField_Response__are_equal(const field_sim__srv__SampleField_Response * lhs, const field_sim__srv__SampleField_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // temperature
  if (lhs->temperature != rhs->temperature) {
    return false;
  }
  // x_enu
  if (lhs->x_enu != rhs->x_enu) {
    return false;
  }
  // y_enu
  if (lhs->y_enu != rhs->y_enu) {
    return false;
  }
  return true;
}

bool
field_sim__srv__SampleField_Response__copy(
  const field_sim__srv__SampleField_Response * input,
  field_sim__srv__SampleField_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // temperature
  output->temperature = input->temperature;
  // x_enu
  output->x_enu = input->x_enu;
  // y_enu
  output->y_enu = input->y_enu;
  return true;
}

field_sim__srv__SampleField_Response *
field_sim__srv__SampleField_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Response * msg = (field_sim__srv__SampleField_Response *)allocator.allocate(sizeof(field_sim__srv__SampleField_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(field_sim__srv__SampleField_Response));
  bool success = field_sim__srv__SampleField_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
field_sim__srv__SampleField_Response__destroy(field_sim__srv__SampleField_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    field_sim__srv__SampleField_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
field_sim__srv__SampleField_Response__Sequence__init(field_sim__srv__SampleField_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Response * data = NULL;

  if (size) {
    data = (field_sim__srv__SampleField_Response *)allocator.zero_allocate(size, sizeof(field_sim__srv__SampleField_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = field_sim__srv__SampleField_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        field_sim__srv__SampleField_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
field_sim__srv__SampleField_Response__Sequence__fini(field_sim__srv__SampleField_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      field_sim__srv__SampleField_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

field_sim__srv__SampleField_Response__Sequence *
field_sim__srv__SampleField_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Response__Sequence * array = (field_sim__srv__SampleField_Response__Sequence *)allocator.allocate(sizeof(field_sim__srv__SampleField_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = field_sim__srv__SampleField_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
field_sim__srv__SampleField_Response__Sequence__destroy(field_sim__srv__SampleField_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    field_sim__srv__SampleField_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
field_sim__srv__SampleField_Response__Sequence__are_equal(const field_sim__srv__SampleField_Response__Sequence * lhs, const field_sim__srv__SampleField_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!field_sim__srv__SampleField_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
field_sim__srv__SampleField_Response__Sequence__copy(
  const field_sim__srv__SampleField_Response__Sequence * input,
  field_sim__srv__SampleField_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(field_sim__srv__SampleField_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    field_sim__srv__SampleField_Response * data =
      (field_sim__srv__SampleField_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!field_sim__srv__SampleField_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          field_sim__srv__SampleField_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!field_sim__srv__SampleField_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "field_sim/srv/detail/sample_field__functions.h"

bool
field_sim__srv__SampleField_Event__init(field_sim__srv__SampleField_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    field_sim__srv__SampleField_Event__fini(msg);
    return false;
  }
  // request
  if (!field_sim__srv__SampleField_Request__Sequence__init(&msg->request, 0)) {
    field_sim__srv__SampleField_Event__fini(msg);
    return false;
  }
  // response
  if (!field_sim__srv__SampleField_Response__Sequence__init(&msg->response, 0)) {
    field_sim__srv__SampleField_Event__fini(msg);
    return false;
  }
  return true;
}

void
field_sim__srv__SampleField_Event__fini(field_sim__srv__SampleField_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  field_sim__srv__SampleField_Request__Sequence__fini(&msg->request);
  // response
  field_sim__srv__SampleField_Response__Sequence__fini(&msg->response);
}

bool
field_sim__srv__SampleField_Event__are_equal(const field_sim__srv__SampleField_Event * lhs, const field_sim__srv__SampleField_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!field_sim__srv__SampleField_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!field_sim__srv__SampleField_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
field_sim__srv__SampleField_Event__copy(
  const field_sim__srv__SampleField_Event * input,
  field_sim__srv__SampleField_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!field_sim__srv__SampleField_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!field_sim__srv__SampleField_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

field_sim__srv__SampleField_Event *
field_sim__srv__SampleField_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Event * msg = (field_sim__srv__SampleField_Event *)allocator.allocate(sizeof(field_sim__srv__SampleField_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(field_sim__srv__SampleField_Event));
  bool success = field_sim__srv__SampleField_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
field_sim__srv__SampleField_Event__destroy(field_sim__srv__SampleField_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    field_sim__srv__SampleField_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
field_sim__srv__SampleField_Event__Sequence__init(field_sim__srv__SampleField_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Event * data = NULL;

  if (size) {
    data = (field_sim__srv__SampleField_Event *)allocator.zero_allocate(size, sizeof(field_sim__srv__SampleField_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = field_sim__srv__SampleField_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        field_sim__srv__SampleField_Event__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
field_sim__srv__SampleField_Event__Sequence__fini(field_sim__srv__SampleField_Event__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      field_sim__srv__SampleField_Event__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

field_sim__srv__SampleField_Event__Sequence *
field_sim__srv__SampleField_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  field_sim__srv__SampleField_Event__Sequence * array = (field_sim__srv__SampleField_Event__Sequence *)allocator.allocate(sizeof(field_sim__srv__SampleField_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = field_sim__srv__SampleField_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
field_sim__srv__SampleField_Event__Sequence__destroy(field_sim__srv__SampleField_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    field_sim__srv__SampleField_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
field_sim__srv__SampleField_Event__Sequence__are_equal(const field_sim__srv__SampleField_Event__Sequence * lhs, const field_sim__srv__SampleField_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!field_sim__srv__SampleField_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
field_sim__srv__SampleField_Event__Sequence__copy(
  const field_sim__srv__SampleField_Event__Sequence * input,
  field_sim__srv__SampleField_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(field_sim__srv__SampleField_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    field_sim__srv__SampleField_Event * data =
      (field_sim__srv__SampleField_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!field_sim__srv__SampleField_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          field_sim__srv__SampleField_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!field_sim__srv__SampleField_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
