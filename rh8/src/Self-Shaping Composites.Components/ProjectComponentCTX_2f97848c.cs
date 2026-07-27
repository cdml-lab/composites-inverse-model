using System;
using SD = System.Drawing;

using Rhino.Geometry;

using Grasshopper.Kernel;
using GH_IO.Serialization;

namespace RhinoCodePlatform.Rhino3D.Projects.Plugin.GH
{
  public sealed class ProjectComponentCTX_2f97848c : ProjectComponent_Base, IGH_ContextualComponent, IGH_DocumentOwner
  {
    static readonly string s_scriptDataId = "2f97848c-393d-45ce-9c70-6c878e69aa9f";
    static readonly string s_scriptIconData = "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABGdBTUEAALGPC/xhBQAAAAlwSFlzAAAOwgAADsIBFShKgAAAAZ5JREFUSEulVUtOwzAQ7RE4AsuMi1RuAEdg63FRu2YDS3ZwhN6A3gDEBajEAeAGdMcKZRKKWBZN4gnuECcueZI3nvF83nw8Gu0BsnBHCPkGzbGWDUbhsqsCYVsdC7daPgilG180xhE+Smeuy/OjE623F7azyYFQUSC8Bw52DqG50W97wYaZazbwPZscfrrxKVnYBIbfCGHVOLHwklyXcgpnYpyseZB7zkiMks3WfOcDeRVHvU68kdo4wlLLGQWaOWcV3rGuZBLe/wFHrCNPRZNJrMM4Kh85cSZa3gemR6hqfS99HqMmBVIjbgotGxGaRR2BmWtZKpieKE2d3hPBHRitoRR4iAN+62leaVl3eolgemsHZqFlv97b0kuEzENrkEGb5nqQUhAOafS9dBIhPGlZH/iviPIvqKKw2drz2GxJvi8cXLYO0A73QL37SGohmXC64a4J069pMfeinzxD9XqWTCAnhOfAaSlRNo4RiGdA2+mEj85Pt/5k4It1/LpexqhLQvXhTI2j4FcjhEetNwgy5d74v5dhFLJjkosY4Adi0YoQBBqFmQAAAABJRU5ErkJggg==";

    public override Guid ComponentGuid { get; } = new Guid("2f97848c-393d-45ce-9c70-6c878e69aa9f");

    public override GH_Exposure Exposure { get; } = GH_Exposure.primary;

    public override bool Obsolete { get; } = false;

    public ProjectComponentCTX_2f97848c() : base(GetResource(s_scriptDataId), s_scriptIconData,
        name: "Inverse_for_website.gh",
        nickname: "Optimize Angles",
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
