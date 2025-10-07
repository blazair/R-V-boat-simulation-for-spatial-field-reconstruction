// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "field_sim/srv/sample_field.hpp"


#ifndef FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__BUILDER_HPP_
#define FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "field_sim/srv/detail/sample_field__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace field_sim
{

namespace srv
{

namespace builder
{

class Init_SampleField_Request_longitude
{
public:
  explicit Init_SampleField_Request_longitude(::field_sim::srv::SampleField_Request & msg)
  : msg_(msg)
  {}
  ::field_sim::srv::SampleField_Request longitude(::field_sim::srv::SampleField_Request::_longitude_type arg)
  {
    msg_.longitude = std::move(arg);
    return std::move(msg_);
  }

private:
  ::field_sim::srv::SampleField_Request msg_;
};

class Init_SampleField_Request_latitude
{
public:
  Init_SampleField_Request_latitude()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SampleField_Request_longitude latitude(::field_sim::srv::SampleField_Request::_latitude_type arg)
  {
    msg_.latitude = std::move(arg);
    return Init_SampleField_Request_longitude(msg_);
  }

private:
  ::field_sim::srv::SampleField_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::field_sim::srv::SampleField_Request>()
{
  return field_sim::srv::builder::Init_SampleField_Request_latitude();
}

}  // namespace field_sim


namespace field_sim
{

namespace srv
{

namespace builder
{

class Init_SampleField_Response_y_enu
{
public:
  explicit Init_SampleField_Response_y_enu(::field_sim::srv::SampleField_Response & msg)
  : msg_(msg)
  {}
  ::field_sim::srv::SampleField_Response y_enu(::field_sim::srv::SampleField_Response::_y_enu_type arg)
  {
    msg_.y_enu = std::move(arg);
    return std::move(msg_);
  }

private:
  ::field_sim::srv::SampleField_Response msg_;
};

class Init_SampleField_Response_x_enu
{
public:
  explicit Init_SampleField_Response_x_enu(::field_sim::srv::SampleField_Response & msg)
  : msg_(msg)
  {}
  Init_SampleField_Response_y_enu x_enu(::field_sim::srv::SampleField_Response::_x_enu_type arg)
  {
    msg_.x_enu = std::move(arg);
    return Init_SampleField_Response_y_enu(msg_);
  }

private:
  ::field_sim::srv::SampleField_Response msg_;
};

class Init_SampleField_Response_temperature
{
public:
  explicit Init_SampleField_Response_temperature(::field_sim::srv::SampleField_Response & msg)
  : msg_(msg)
  {}
  Init_SampleField_Response_x_enu temperature(::field_sim::srv::SampleField_Response::_temperature_type arg)
  {
    msg_.temperature = std::move(arg);
    return Init_SampleField_Response_x_enu(msg_);
  }

private:
  ::field_sim::srv::SampleField_Response msg_;
};

class Init_SampleField_Response_success
{
public:
  Init_SampleField_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SampleField_Response_temperature success(::field_sim::srv::SampleField_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SampleField_Response_temperature(msg_);
  }

private:
  ::field_sim::srv::SampleField_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::field_sim::srv::SampleField_Response>()
{
  return field_sim::srv::builder::Init_SampleField_Response_success();
}

}  // namespace field_sim


namespace field_sim
{

namespace srv
{

namespace builder
{

class Init_SampleField_Event_response
{
public:
  explicit Init_SampleField_Event_response(::field_sim::srv::SampleField_Event & msg)
  : msg_(msg)
  {}
  ::field_sim::srv::SampleField_Event response(::field_sim::srv::SampleField_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::field_sim::srv::SampleField_Event msg_;
};

class Init_SampleField_Event_request
{
public:
  explicit Init_SampleField_Event_request(::field_sim::srv::SampleField_Event & msg)
  : msg_(msg)
  {}
  Init_SampleField_Event_response request(::field_sim::srv::SampleField_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_SampleField_Event_response(msg_);
  }

private:
  ::field_sim::srv::SampleField_Event msg_;
};

class Init_SampleField_Event_info
{
public:
  Init_SampleField_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SampleField_Event_request info(::field_sim::srv::SampleField_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_SampleField_Event_request(msg_);
  }

private:
  ::field_sim::srv::SampleField_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::field_sim::srv::SampleField_Event>()
{
  return field_sim::srv::builder::Init_SampleField_Event_info();
}

}  // namespace field_sim

#endif  // FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__BUILDER_HPP_
