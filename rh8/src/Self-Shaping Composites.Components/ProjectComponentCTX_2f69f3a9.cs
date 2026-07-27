using System;
using SD = System.Drawing;

using Rhino.Geometry;

using Grasshopper.Kernel;
using GH_IO.Serialization;

namespace RhinoCodePlatform.Rhino3D.Projects.Plugin.GH
{
  public sealed class ProjectComponentCTX_2f69f3a9 : ProjectComponent_Base, IGH_ContextualComponent, IGH_DocumentOwner
  {
    static readonly string s_scriptDataId = "2f69f3a9-99e9-4085-9300-2d646ed84ab8";
    static readonly string s_scriptIconData = "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";

    public override Guid ComponentGuid { get; } = new Guid("2f69f3a9-99e9-4085-9300-2d646ed84ab8");

    public override GH_Exposure Exposure { get; } = GH_Exposure.primary;

    public override bool Obsolete { get; } = false;

    public ProjectComponentCTX_2f69f3a9() : base(GetResource(s_scriptDataId), s_scriptIconData,
        name: "angles-graphics.gh",
        nickname: "Preview",
        description: @"",
        category: "Self-Shaping Composites",
        subCategory: "Default"
        )
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager _) { }
    protected override void RegisterOutputParams(GH_OutputParamManager _) { }

    protected override void BeforeSolveInstance() => m_script.BeforeSolve(this);

    protected override void SolveInstance(IGH_DataAccess DA) => m_script.Solve(this, DA);

    protected override void AfterSolveInstance() => m_script.AfterSolve(this);

    public override BoundingBox ClippingBox => m_script.GetClipBox(this);

    public override void DrawViewportWires(IGH_PreviewArgs args) => m_script.DrawWires(this, args);

    public override void DrawViewportMeshes(IGH_PreviewArgs args) => m_script.DrawMeshes(this, args);

    #region IGH_ContextualComponent
    GH_Archive IGH_ContextualComponent.Archive => m_script.Archive;
    #endregion

    #region IGH_DocumentOwner
    GH_Document IGH_DocumentOwner.OwnerDocument() => OnPingDocument();

    void IGH_DocumentOwner.DocumentClosed(GH_Document document)
    {
      // Internal docs are embedded, we don't need to do anything.
    }

    void IGH_DocumentOwner.DocumentModified(GH_Document document)
    {
      // Internal docs are embedded, we don't need to do anything.
    }
    #endregion
  }
}
