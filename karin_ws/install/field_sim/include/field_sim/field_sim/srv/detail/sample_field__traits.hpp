// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from field_sim:srv/SampleField.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "field_sim/srv/sample_field.hpp"


#ifndef FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__TRAITS_HPP_
#define FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "field_sim/srv/detail/sample_field__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace field_sim
{

namespace srv
{

inline void to_flow_style_yaml(
  const SampleField_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: latitude
  {
    out << "latitude: ";
    rosidl_generator_traits::value_to_yaml(msg.latitude, out);
    out << ", ";
  }

  // member: longitude
  {
    out << "longitude: ";
    rosidl_generator_traits::value_to_yaml(msg.longitude, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SampleField_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: latitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "latitude: ";
    rosidl_generator_traits::value_to_yaml(msg.latitude, out);
    out << "\n";
  }

  // member: longitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "longitude: ";
    rosidl_generator_traits::value_to_yaml(msg.longitude, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SampleField_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace field_sim

namespace rosidl_generator_traits
{

[[deprecated("use field_sim::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const field_sim::srv::SampleField_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  field_sim::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use field_sim::srv::to_yaml() instead")]]
inline std::string to_yaml(const field_sim::srv::SampleField_Request & msg)
{
  return field_sim::srv::to_yaml(msg);
}

template<>
inline const char * data_type<field_sim::srv::SampleField_Request>()
{
  return "field_sim::srv::SampleField_Request";
}

template<>
inline const char * name<field_sim::srv::SampleField_Request>()
{
  return "field_sim/srv/SampleField_Request";
}

template<>
struct has_fixed_size<field_sim::srv::SampleField_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<field_sim::srv::SampleField_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<field_sim::srv::SampleField_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace field_sim
{

namespace srv
{

inline void to_flow_style_yaml(
  const SampleField_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: temperature
  {
    out << "temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.temperature, out);
    out << ", ";
  }

  // member: x_enu
  {
    out << "x_enu: ";
    rosidl_generator_traits::value_to_yaml(msg.x_enu, out);
    out << ", ";
  }

  // member: y_enu
  {
    out << "y_enu: ";
    rosidl_generator_traits::value_to_yaml(msg.y_enu, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SampleField_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.temperature, out);
    out << "\n";
  }

  // member: x_enu
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x_enu: ";
    rosidl_generator_traits::value_to_yaml(msg.x_enu, out);
    out << "\n";
  }

  // member: y_enu
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y_enu: ";
    rosidl_generator_traits::value_to_yaml(msg.y_enu, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SampleField_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace field_sim

namespace rosidl_generator_traits
{

[[deprecated("use field_sim::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const field_sim::srv::SampleField_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  field_sim::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use field_sim::srv::to_yaml() instead")]]
inline std::string to_yaml(const field_sim::srv::SampleField_Response & msg)
{
  return field_sim::srv::to_yaml(msg);
}

template<>
inline const char * data_type<field_sim::srv::SampleField_Response>()
{
  return "field_sim::srv::SampleField_Response";
}

template<>
inline const char * name<field_sim::srv::SampleField_Response>()
{
  return "field_sim/srv/SampleField_Response";
}

template<>
struct has_fixed_size<field_sim::srv::SampleField_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<field_sim::srv::SampleField_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<field_sim::srv::SampleField_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace field_sim
{

namespace srv
{

inline void to_flow_style_yaml(
  const SampleField_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SampleField_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SampleField_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace field_sim

namespace rosidl_generator_traits
{

[[deprecated("use field_sim::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const field_sim::srv::SampleField_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  field_sim::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use field_sim::srv::to_yaml() instead")]]
inline std::string to_yaml(const field_sim::srv::SampleField_Event & msg)
{
  return field_sim::srv::to_yaml(msg);
}

template<>
inline const char * data_type<field_sim::srv::SampleField_Event>()
{
  return "field_sim::srv::SampleField_Event";
}

template<>
inline const char * name<field_sim::srv::SampleField_Event>()
{
  return "field_sim/srv/SampleField_Event";
}

template<>
struct has_fixed_size<field_sim::srv::SampleField_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<field_sim::srv::SampleField_Event>
  : std::integral_constant<bool, has_bounded_size<field_sim::srv::SampleField_Request>::value && has_bounded_size<field_sim::srv::SampleField_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<field_sim::srv::SampleField_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<field_sim::srv::SampleField>()
{
  return "field_sim::srv::SampleField";
}

template<>
inline const char * name<field_sim::srv::SampleField>()
{
  return "field_sim/srv/SampleField";
}

template<>
struct has_fixed_size<field_sim::srv::SampleField>
  : std::integral_constant<
    bool,
    has_fixed_size<field_sim::srv::SampleField_Request>::value &&
    has_fixed_size<field_sim::srv::SampleField_Response>::value
  >
{
};

template<>
struct has_bounded_size<field_sim::srv::SampleField>
  : std::integral_constant<
    bool,
    has_bounded_size<field_sim::srv::SampleField_Request>::value &&
    has_bounded_size<field_sim::srv::SampleField_Response>::value
  >
{
};

template<>
struct is_service<field_sim::srv::SampleField>
  : std::true_type
{
};

template<>
struct is_service_request<field_sim::srv::SampleField_Request>
  : std::true_type
{
};

template<>
struct is_service_response<field_sim::srv::SampleField_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // FIELD_SIM__SRV__DETAIL__SAMPLE_FIELD__TRAITS_HPP_
