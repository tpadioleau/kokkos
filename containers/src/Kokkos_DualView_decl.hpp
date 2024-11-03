#ifndef KOKKOS_DUALVIEW_HPP
#define KOKKOS_DUALVIEW_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_DUALVIEW
#endif

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Error.hpp>

namespace Kokkos {

template <class DataType, class... Properties>
class DualView;

template <class>
struct is_dual_view : public std::false_type {};

template <class T>
inline constexpr bool is_dual_view_v = is_dual_view<T>::value;

template <class DataType, class... Properties>
class DualView : public ViewTraits<DataType, Properties...> {
  template <class, class...>
  friend class DualView;

 public:
  using traits = ViewTraits<DataType, Properties...>;

  using host_mirror_space = typename traits::host_mirror_space;

  using t_dev = View<typename traits::data_type, Properties...>;

  using t_host = typename t_dev::HostMirror;

  using t_dev_const = View<typename traits::const_data_type, Properties...>;

  using t_host_const = typename t_dev_const::HostMirror;

  using t_dev_const_randomread =
      View<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type,
           Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  using t_host_const_randomread = typename t_dev_const_randomread::HostMirror;

  using t_dev_um =
      View<typename traits::data_type, typename traits::array_layout,
           typename traits::device_type, MemoryUnmanaged>;

  using t_host_um =
      View<typename t_host::data_type, typename t_host::array_layout,
           typename t_host::device_type, MemoryUnmanaged>;

  using t_dev_const_um =
      View<typename traits::const_data_type, typename traits::array_layout,
           typename traits::device_type, MemoryUnmanaged>;

  using t_host_const_um =
      View<typename t_host::const_data_type, typename t_host::array_layout,
           typename t_host::device_type, MemoryUnmanaged>;

  using t_dev_const_randomread_um =
      View<typename t_host::const_data_type, typename t_host::array_layout,
           typename t_host::device_type,
           Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;

  using t_host_const_randomread_um =
      typename t_dev_const_randomread_um::HostMirror;

 protected:
  using t_modified_flags = View<unsigned int[2], LayoutLeft, Kokkos::HostSpace>;
  t_modified_flags modified_flags;

 public:
  t_dev d_view;
  t_host h_view;

  DualView() = default;

  DualView(const std::string& label,
           const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  template <class... P>
  DualView(const Impl::ViewCtorProp<P...>& arg_prop,
           std::enable_if_t<!Impl::ViewCtorProp<P...>::has_pointer,
                            size_t> const n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7                   = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  template <typename DT, typename... DP>
  DualView(const DualView<DT, DP...>& src);

  template <class DT, class... DP, class Arg0, class... Args>
  DualView(const DualView<DT, DP...>& src, const Arg0& arg0, Args... args);

  DualView(const t_dev& d_view_, const t_host& h_view_);

  struct impl_dualview_is_single_device;

  template <typename Device>
  struct impl_device_matches_tdev_device;

  template <typename Device>
  struct impl_device_matches_thost_device;

  template <typename Device>
  struct impl_device_matches_thost_exec;

  template <typename Device>
  struct impl_device_matches_tdev_exec;

  template <typename Device>
  struct impl_device_matches_tdev_memory_space;

  template <class Device>
  KOKKOS_FUNCTION auto view() const;

  KOKKOS_INLINE_FUNCTION
  t_host view_host() const;

  KOKKOS_INLINE_FUNCTION
  t_dev view_device() const;

  KOKKOS_INLINE_FUNCTION constexpr bool is_allocated();

  template <class Device>
  static int get_device_side();

  static constexpr const int view_header_size = 128;

  void impl_report_host_sync() const noexcept;
  void impl_report_device_sync() const noexcept;

  template <class Device, class... Args>
  void sync_impl(std::true_type, Args const&... args);

  template <class Device>
  void sync(const std::enable_if_t<
                (std::is_same_v<typename traits::data_type,
                                typename traits::non_const_data_type>) ||
                    (std::is_same_v<Device, int>),
                int>& = 0);

  template <class Device, class ExecutionSpace>
  void sync(const ExecutionSpace& exec,
            const std::enable_if_t<
                (std::is_same_v<typename traits::data_type,
                                typename traits::non_const_data_type>) ||
                    (std::is_same_v<Device, int>),
                int>& = 0);

  template <class Device, class... Args>
  void sync_impl(std::false_type, Args const&...);

  template <class Device>
  void sync(const std::enable_if_t<
                (!std::is_same_v<typename traits::data_type,
                                 typename traits::non_const_data_type>) ||
                    (std::is_same_v<Device, int>),
                int>& = 0);

  template <class Device, class ExecutionSpace>
  void sync(const ExecutionSpace& exec,
            const std::enable_if_t<
                (!std::is_same_v<typename traits::data_type,
                                 typename traits::non_const_data_type>) ||
                    (std::is_same_v<Device, int>),
                int>& = 0);

  template <typename... Args>
  void sync_host_impl(Args const&... args);

  template <class ExecSpace>
  void sync_host(const ExecSpace& exec);
  void sync_host();

  template <typename... Args>
  void sync_device_impl(Args const&... args);

  template <class ExecSpace>
  void sync_device(const ExecSpace& exec);
  void sync_device();

  template <class Device>
  bool need_sync() const;

  inline bool need_sync_host() const;

  inline bool need_sync_device() const;
  void impl_report_device_modification();
  void impl_report_host_modification();

  template <class Device, class Dummy = DualView,
            std::enable_if_t<!Dummy::impl_dualview_is_single_device::value>* =
                nullptr>
  void modify();

  template <
      class Device, class Dummy = DualView,
      std::enable_if_t<Dummy::impl_dualview_is_single_device::value>* = nullptr>
  void modify();

  template <class Dummy = DualView,
            std::enable_if_t<!Dummy::impl_dualview_is_single_device::value>* =
                nullptr>
  inline void modify_host();

  template <
      class Dummy = DualView,
      std::enable_if_t<Dummy::impl_dualview_is_single_device::value>* = nullptr>
  inline void modify_host();

  template <class Dummy = DualView,
            std::enable_if_t<!Dummy::impl_dualview_is_single_device::value>* =
                nullptr>
  inline void modify_device();

  template <
      class Dummy = DualView,
      std::enable_if_t<Dummy::impl_dualview_is_single_device::value>* = nullptr>
  inline void modify_device();

  inline void clear_sync_state();

  template <class... ViewCtorArgs>
  void impl_realloc(const size_t n0, const size_t n1, const size_t n2,
                    const size_t n3, const size_t n4, const size_t n5,
                    const size_t n6, const size_t n7,
                    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop);

  template <class... ViewCtorArgs>
  void realloc(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
               const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  void realloc(const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  template <typename I>
  std::enable_if_t<Impl::is_view_ctor_property<I>::value> realloc(
      const I& arg_prop, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  template <class... ViewCtorArgs>
  void impl_resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
                   const size_t n0, const size_t n1, const size_t n2,
                   const size_t n3, const size_t n4, const size_t n5,
                   const size_t n6, const size_t n7);

 private:
  template <class... ViewCtorArgs>
  inline void resync_host(Impl::ViewCtorProp<ViewCtorArgs...> const&);

  template <class... ViewCtorArgs>
  inline void resync_device(Impl::ViewCtorProp<ViewCtorArgs...> const&);

 public:
  void resize(const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  template <class... ViewCtorArgs>
  void resize(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
              const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  template <class I>
  std::enable_if_t<Impl::is_view_ctor_property<I>::value> resize(
      const I& arg_prop, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  KOKKOS_INLINE_FUNCTION constexpr size_t span() const;

  KOKKOS_INLINE_FUNCTION bool span_is_contiguous() const;

  template <typename iType>
  void stride(iType* stride_) const;

  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<std::is_integral_v<iType>,
                                                    size_t>
  extent(const iType& r) const;

  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<std::is_integral_v<iType>,
                                                    int>
  extent_int(const iType& r) const;
};

template <class DataType, class... Properties, class... Args>
auto subview(const DualView<DataType, Properties...>& src, Args&&... args);

template <class DT, class... DP, class ST, class... SP>
void deep_copy(DualView<DT, DP...>& dst, const DualView<ST, SP...>& src);

template <class ExecutionSpace, class DT, class... DP, class ST, class... SP>
void deep_copy(const ExecutionSpace& exec, DualView<DT, DP...>& dst,
               const DualView<ST, SP...>& src);

template <class... Properties, class... Args>
void resize(DualView<Properties...>& dv, Args&&... args) noexcept(
    noexcept(dv.resize(std::forward<Args>(args)...)));

template <class... ViewCtorArgs, class... Properties, class... Args>
void resize(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    DualView<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.resize(arg_prop,
                                                std::forward<Args>(args)...)));

template <class I, class... Properties, class... Args>
std::enable_if_t<Impl::is_view_ctor_property<I>::value> resize(
    const I& arg_prop, DualView<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.resize(arg_prop,
                                                std::forward<Args>(args)...)));

template <class... ViewCtorArgs, class... Properties, class... Args>
void realloc(
    const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
    DualView<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.realloc(std::forward<Args>(args)...)));

template <class... Properties, class... Args>
void realloc(DualView<Properties...>& dv, Args&&... args) noexcept;

template <class I, class... Properties, class... Args>
std::enable_if_t<Impl::is_view_ctor_property<I>::value> realloc(
    const I& arg_prop, DualView<Properties...>& dv,
    Args&&... args) noexcept(noexcept(dv.realloc(arg_prop,
                                                 std::forward<Args>(args)...)));

}  // end namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_DUALVIEW
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_DUALVIEW
#endif
#endif
